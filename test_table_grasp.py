"""Table-top grasp test: 1m table + red block, robot at x=-0.9, multi-angle pick."""
import sys, os, numpy as np, torch, cv2
sys.path.insert(0, os.path.dirname(__file__))
import tidyverse_agent, mani_skill.envs, gymnasium as gym
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mplib.collision_detection.fcl import *
from sapien.physx import *
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv
import sapien
import argparse
from mani_skill.utils.wrappers.record import RecordEpisode

# --- Monkey-patch for Robotiq scaled convex meshes ---
@staticmethod
def _pc(comp):
    shapes, shape_poses = [], []
    for shape in comp.collision_shapes:
        shape_poses.append(MPPose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox): c_geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            c_geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0): verts = verts * np.array(shape.scale)
            c_geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeSphere): c_geom = Sphere(radius=shape.radius)
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            c_geom = BVHModel(); c_geom.begin_model()
            c_geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles); c_geom.end_model()
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]; d = n.dot(shape_poses[-1].p)
            c_geom = Halfspace(n=n, d=d); shape_poses[-1] = MPPose()
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            c_geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        else: continue
        shapes.append(CollisionObject(c_geom))
    if not shapes: return None
    return FCLObject(comp.name if isinstance(comp, PhysxArticulationLinkComponent) else _conv.convert_object_name(comp.entity), comp.entity.pose, shapes, shape_poses)
SapienPlanningWorld.convert_physx_component = _pc

SDK_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', default='human', choices=['human', 'rgb_array'],
                        help='human=GUI window, rgb_array=save video')
    parser.add_argument('--robot-x', type=float, default=-0.3, help='Robot root X position')
    parser.add_argument('--table-x', type=float, default=0.0, help='Table center X')
    parser.add_argument('--table-height', type=float, default=0.762, help='Table height in meters (default: 30 in)')
    args = parser.parse_args()

    # Raise camera 0.7m higher than default (eye z: 0.6 -> 1.3)
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.sensors.camera import CameraConfig
    @property
    def _raised_cam(self):
        from mani_skill.utils import sapien_utils as ms_sapien_utils
        pose = ms_sapien_utils.look_at(eye=[0.9, 1.0, 1.5], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    PickCubeEnv._default_human_render_camera_configs = _raised_cam

    render_mode = args.render if args.render == 'human' else 'rgb_array'
    env = gym.make('PickCube-v1', num_envs=1, robot_uids='tidyverse',
                   control_mode='whole_body', render_mode=render_mode)
    video_dir = os.path.join(os.path.dirname(__file__), 'videos')
    if render_mode == 'rgb_array':
        env = RecordEpisode(env, output_dir=video_dir, save_video=True,
                            max_steps_per_video=10000, video_fps=30)
    obs, info = env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    scene_ms = env.unwrapped.scene
    scene = scene_ms.sub_scenes[0]

    robot.set_pose(sapien.Pose(p=[args.robot_x, 0, 0]))

    # Table
    tb = scene_ms.create_actor_builder()
    tb.add_box_collision(half_size=[0.3, 0.3, args.table_height/2])
    tb.add_box_visual(half_size=[0.3, 0.3, args.table_height/2],
                      material=sapien.render.RenderMaterial(base_color=[0.6, 0.4, 0.2, 1.0]))
    table = tb.build_static(name="table")
    table.set_pose(sapien.Pose(p=[args.table_x, 0, args.table_height/2]))

    # Block on table
    bb = scene_ms.create_actor_builder()
    bb.add_box_collision(half_size=[0.02, 0.02, 0.02])
    bb.add_box_visual(half_size=[0.02, 0.02, 0.02],
                      material=sapien.render.RenderMaterial(base_color=[1.0, 0.2, 0.2, 1.0]))
    block = bb.build(name="red_block")
    block.set_pose(sapien.Pose(p=[args.table_x, 0, args.table_height + 0.04]))

    base_cmd = np.array([args.robot_x, 0.0, 0.0])  # base x, y, yaw position target
    hold = torch.tensor(np.concatenate([SDK_HOME, [0.0], base_cmd]), dtype=torch.float32).unsqueeze(0)
    is_human = (render_mode == 'human')

    def step(action):
        obs = env.step(action)
        if is_human: env.render()
        return obs

    def wait_until_stable(env, hold, max_steps=300, vel_thresh=1e-3, window=10):
        """Step until robot qvel is small for `window` consecutive steps."""
        stable_count = 0
        for si in range(max_steps):
            step(hold)
            qvel = robot.get_qvel().cpu().numpy()[0]
            if np.max(np.abs(qvel)) < vel_thresh:
                stable_count += 1
                if stable_count >= window:
                    print(f"    Stabilized after {si+1} steps")
                    return si + 1
            else:
                stable_count = 0
        print(f"    WARNING: not fully stable after {max_steps} steps (max |qvel|={np.max(np.abs(qvel)):.4f})")
        return max_steps

    print("Waiting for robot to stabilize at initial pose...")
    wait_until_stable(env, hold)

    arm_base = next(l for l in robot.get_links() if l.get_name()=='panda_link0').pose.p[0].cpu().numpy()
    block_pos = block.pose.p[0].cpu().numpy()
    dist = np.linalg.norm(arm_base[:2] - block_pos[:2])
    print(f"Arm base:     {arm_base}")
    print(f"Block:        {block_pos}")
    print(f"Dist XY:      {dist:.3f}m")
    print(f"Block height: {block_pos[2]:.3f}m")

    # Planner
    pw = SapienPlanningWorld(scene, [robot._objs[0]])
    eef = [l for l in pw.get_planned_articulations()[0].get_pinocchio_model().get_link_names() if 'eef' in l][0]
    planner = SapienPlanner(pw, move_group=eef)
    mask_arm = np.array([True]*3 + [False]*7 + [True]*6)
    mask_wb  = np.array([False]*3 + [False]*7 + [True]*6)

    block_pos = block.pose.p[0].cpu().numpy()
    approach_yaw = np.arctan2(block_pos[1]-arm_base[1], block_pos[0]-arm_base[0])

    # EEF frame is at the TCP (between finger pads, 14.5cm from gripper base).
    # Block half-size is 2cm, so target EEF at block center for a centered grasp.
    grasps = [
        ('Top-Down', block_pos+[0,0,0.0], [0,1,0,0]),
        ('Front', block_pos+[-0.0*np.cos(approach_yaw),-0.0*np.sin(approach_yaw),0.0],
         list(R.from_euler('yz',[np.pi/2,approach_yaw]).as_quat()[[3,0,1,2]])),
        ('Angled45', block_pos+[-0.02*np.cos(approach_yaw),-0.02*np.sin(approach_yaw),0.02],
         list(euler2quat(0,3*np.pi/4,approach_yaw))),
    ]

    def diagnose_failure(label, result, target_pose, current_qpos, planner, pw, mask=None):
        """Print detailed diagnostics when planning fails."""
        print(f"  {label}: FAILED — {result['status']}")
        print(f"    Target pos:  {np.array(target_pose.p)}")
        print(f"    Target quat: {np.array(target_pose.q)}")
        print(f"    Current qpos (arm): {current_qpos[3:10]}")
        # List obstacles in planning world
        obj_names = pw.get_object_names()
        print(f"    Obstacles ({len(obj_names)}):")
        for oname in obj_names:
            obj = pw.get_object(oname)
            print(f"      - {oname}  pose={obj.pose}")
        # Check current-state collisions
        cur_collisions = pw.check_collision()
        if cur_collisions:
            print(f"    Current-state collisions ({len(cur_collisions)}):")
            for c in cur_collisions:
                print(f"      - {c.link_name1}({c.object_name1}) <-> {c.link_name2}({c.object_name2})")
        # Use planner.IK() which handles full qpos, mask, and collision checking
        try:
            ik_status, ik_solutions = planner.IK(
                target_pose, current_qpos, mask=mask, n_init_qpos=40, verbose=True
            )
            if ik_solutions is not None:
                print(f"    IK check: found {len(ik_solutions)} solution(s) — path planning (RRT) failed to find collision-free route")
                for i, q in enumerate(ik_solutions):
                    print(f"      solution {i}: arm_qpos={q[3:10]}")
            else:
                print(f"    IK check: {ik_status}")
                # Also try without mask to see if it's a reachability vs mask issue
                ik_status2, ik_solutions2 = planner.IK(
                    target_pose, current_qpos, mask=None, n_init_qpos=40, verbose=True
                )
                if ik_solutions2 is not None:
                    print(f"    IK check (no mask): found {len(ik_solutions2)} solution(s) — mask is too restrictive")
                else:
                    print(f"    IK check (no mask): {ik_status2} — pose is unreachable")
        except Exception as e:
            print(f"    IK check error: {e}")

    for gi, (name, target_p, target_q) in enumerate(grasps):
        print(f"\n--- [{gi+1}/{len(grasps)}] {name} ---")
        print(f"  Grasp target: pos={np.array(target_p)}  quat={np.array(target_q)}")
        # Reset arm and wait for stabilization
        qpos = robot.get_qpos().cpu().numpy()[0]
        qpos[3:10] = SDK_HOME; qpos[10:] = 0.0
        robot.set_qpos(torch.tensor(qpos, dtype=torch.float32).unsqueeze(0))
        print("  Waiting for arm reset to stabilize...")
        wait_until_stable(env, hold, max_steps=200)
        try: planner.update_from_simulation()
        except: pass

        cq = robot.get_qpos().cpu().numpy()[0]
        print(f"  Actual base pos: {cq[:3]}  arm qpos: {cq[3:10]}")
        # Pre-grasp (8cm above target)
        pre_p = np.array(target_p) + [0, 0, 0.08]
        pre_pose = MPPose(p=pre_p, q=np.array(target_q))
        for mode, m in [("arm-only", mask_arm), ("whole-body", mask_wb)]:
            r = planner.plan_pose(pre_pose, cq, mask=m, planning_time=5.0)
            if r['status'] == 'Success':
                print(f"  Pre-grasp ({mode}): OK  ({r['position'].shape[0]} waypoints, {r['duration']:.2f}s)")
                used_mask = m; break
            diagnose_failure(f"Pre-grasp ({mode})", r, pre_pose, cq, planner, pw, mask=m)
        else:
            print(f"  SKIPPED — no solution for any mode"); continue

        # Execute pre-grasp
        for i in range(r['position'].shape[0]):
            act = np.concatenate([r['position'][i, 3:10], [0.0], base_cmd])
            step(torch.tensor(act, dtype=torch.float32).unsqueeze(0))

        # Approach
        try: planner.update_from_simulation()
        except: pass
        cq = robot.get_qpos().cpu().numpy()[0]
        approach_pose = MPPose(p=np.array(target_p), q=np.array(target_q))
        r2 = planner.plan_pose(approach_pose, cq, mask=used_mask, planning_time=5.0)
        if r2['status'] == 'Success':
            print(f"  Approach: OK  ({r2['position'].shape[0]} waypoints, {r2['duration']:.2f}s)")
            for i in range(r2['position'].shape[0]):
                act = np.concatenate([r2['position'][i, 3:10], [0.0], base_cmd])
                step(torch.tensor(act, dtype=torch.float32).unsqueeze(0))
        else:
            diagnose_failure("Approach", r2, approach_pose, cq, planner, pw, mask=used_mask)

        # Close gripper
        aq = robot.get_qpos().cpu().numpy()[0][3:10]
        for _ in range(30):
            act = np.concatenate([aq, [0.81], base_cmd])
            step(torch.tensor(act, dtype=torch.float32).unsqueeze(0))

        # Lift
        try: planner.update_from_simulation()
        except: pass
        cq = robot.get_qpos().cpu().numpy()[0]
        lift_pose = MPPose(p=np.array(target_p)+[0,0,0.15], q=np.array(target_q))
        r3 = planner.plan_pose(lift_pose, cq, mask=used_mask, planning_time=3.0)
        if r3['status'] == 'Success':
            print(f"  Lift: OK  ({r3['position'].shape[0]} waypoints, {r3['duration']:.2f}s)")
            for i in range(r3['position'].shape[0]):
                act = np.concatenate([r3['position'][i, 3:10], [0.81], base_cmd])
                step(torch.tensor(act, dtype=torch.float32).unsqueeze(0))
        else:
            diagnose_failure("Lift", r3, lift_pose, cq, planner, pw, mask=used_mask)

        # Hold
        for _ in range(60):
            step(torch.tensor(act, dtype=torch.float32).unsqueeze(0))

    if render_mode == 'human':
        print("\nDone! Close the window to exit.")
        while True:
            env.step(hold)
            env.render()
    else:
        env.close()
        print(f"\nDone! Video saved to {video_dir}/")

if __name__ == '__main__':
    main()
