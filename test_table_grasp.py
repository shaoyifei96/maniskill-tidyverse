"""Table-top grasp test: table + red block, multi-angle pick with diagnostics."""
import sys
import os
import argparse
import numpy as np
import torch
import cv2
import sapien
import gymnasium as gym

import maniskill_tidyverse.tidyverse_agent  # noqa: F401 — registers 'tidyverse' robot
import mani_skill.envs  # noqa: F401 — registers ManiSkill envs

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib.collision_detection.fcl import (
    Box, Capsule, Convex, Sphere, BVHModel, Halfspace, Cylinder,
    CollisionObject, FCLObject,
)
from sapien.physx import (
    PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh, PhysxCollisionShapeSphere,
    PhysxCollisionShapeTriangleMesh, PhysxCollisionShapePlane,
    PhysxCollisionShapeCylinder, PhysxArticulationLinkComponent,
)
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.sensors.camera import CameraConfig
from maniskill_tidyverse.viz_planning_world import save_planning_world

# ─── Constants ────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81   # Robotiq 85 joint range [0, 0.81] rad
PRE_GRASP_HEIGHT = 0.08  # metres above grasp target
LIFT_HEIGHT = 0.15        # metres above grasp target

# Planning masks: True = locked joint, False = free joint
# Layout: [base_x, base_y, base_yaw, arm×7, gripper×6]
MASK_ARM_ONLY = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)


# ─── Monkey-patch: apply scale to Robotiq convex collision meshes ─────────────

@staticmethod
def _convert_physx_component(comp):
    shapes, shape_poses = [], []
    for shape in comp.collision_shapes:
        shape_poses.append(MPPose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox):
            geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0):
                verts = verts * np.array(shape.scale)
            geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeSphere):
            geom = Sphere(radius=shape.radius)
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            geom = BVHModel()
            geom.begin_model()
            geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)
            geom.end_model()
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]
            d = n.dot(shape_poses[-1].p)
            geom = Halfspace(n=n, d=d)
            shape_poses[-1] = MPPose()
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        else:
            continue
        shapes.append(CollisionObject(geom))
    if not shapes:
        return None
    name = (comp.name if isinstance(comp, PhysxArticulationLinkComponent)
            else _conv.convert_object_name(comp.entity))
    return FCLObject(name, comp.entity.pose, shapes, shape_poses)

SapienPlanningWorld.convert_physx_component = _convert_physx_component


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_recorder(env):
    """Walk the wrapper chain to find RecordEpisode (has render_images)."""
    rec = env
    while hasattr(rec, 'env'):
        if hasattr(rec, 'render_images'):
            return rec
        rec = rec.env
    if hasattr(rec, 'render_images'):
        return rec
    return None


def _get_frame_count(env):
    rec = _find_recorder(env)
    return len(rec.render_images) if rec else 0


def overlay_text_on_frames(env, text, n_frames=None):
    """Burn text onto the last n_frames of the recorded video."""
    rec = _find_recorder(env)
    if not rec or not rec.render_images:
        return
    frames = rec.render_images
    start = max(0, len(frames) - n_frames) if n_frames else 0
    for i in range(start, len(frames)):
        img = frames[i]
        if not isinstance(img, np.ndarray):
            continue
        img = img.copy()
        h = img.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600)
        thickness = max(1, int(h / 300))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, th + 10
        cv2.rectangle(img, (x - 5, y - th - 5),
                      (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
        frames[i] = img


def pause_with_label(env, step_fn, hold_action, label, fps=30, seconds=1.0):
    """Hold current pose for `seconds`, overlaying label text on frames."""
    n_steps = int(fps * seconds)
    start_fc = _get_frame_count(env)
    for _ in range(n_steps):
        step_fn(hold_action)
    overlay_text_on_frames(env, label, n_frames=_get_frame_count(env) - start_fc)


def make_action(arm_qpos, gripper, base_cmd):
    """Build a single action tensor: [arm(7), gripper(1), base(3)]."""
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


def sync_planner(planner):
    """Re-sync planner state from simulation, ignoring errors."""
    try:
        planner.update_from_simulation()
    except Exception:
        pass


def get_robot_qpos(robot):
    """Get the current full qpos as a 1-D numpy array."""
    return robot.get_qpos().cpu().numpy()[0]


def wait_until_stable(step_fn, hold, robot, max_steps=300,
                      vel_thresh=1e-3, window=10):
    """Step simulation until robot velocities settle."""
    stable_count = 0
    for si in range(max_steps):
        step_fn(hold)
        qvel = robot.get_qvel().cpu().numpy()[0]
        if np.max(np.abs(qvel)) < vel_thresh:
            stable_count += 1
            if stable_count >= window:
                print(f"    Stabilized after {si + 1} steps")
                return si + 1
        else:
            stable_count = 0
    print(f"    WARNING: not stable after {max_steps} steps "
          f"(max |qvel|={np.max(np.abs(qvel)):.4f})")
    return max_steps


def check_joint_limits(qpos, joint_limits, joint_names, label=""):
    """Print warnings for any joints within 2% of their limits."""
    qi = 0
    for limits, name in zip(joint_limits, joint_names):
        if limits.ndim == 2:
            for d in range(limits.shape[0]):
                if qi >= len(qpos):
                    return
                lo, hi = limits[d, 0], limits[d, 1]
                margin = (hi - lo) * 0.02
                if qpos[qi] <= lo + margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {qpos[qi]:.4f} "
                          f"at lower limit {lo:.4f}")
                elif qpos[qi] >= hi - margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {qpos[qi]:.4f} "
                          f"at upper limit {hi:.4f}")
                qi += 1
        elif limits.ndim == 1 and limits.shape[0] >= 2:
            if qi >= len(qpos):
                return
            lo, hi = limits[0], limits[1]
            margin = (hi - lo) * 0.02
            if qpos[qi] <= lo + margin:
                print(f"    JOINT LIMIT {label}: {name} = {qpos[qi]:.4f} "
                      f"at lower limit {lo:.4f}")
            elif qpos[qi] >= hi - margin:
                print(f"    JOINT LIMIT {label}: {name} = {qpos[qi]:.4f} "
                      f"at upper limit {hi:.4f}")
            qi += 1
        else:
            qi += 1


def execute_trajectory(traj, step_fn, gripper, lock_base=False,
                       env=None, label=None, robot=None,
                       settle_thresh=0.01, settle_steps=100):
    """Execute a planned trajectory, sending arm + base targets each step.

    If lock_base is True, clamp base to the first waypoint's values
    (RRT may drift base even with arm-only IK mask).
    After the trajectory, holds the final target until joints converge.
    """
    base_cmd = traj[0, 0:3] if lock_base else None
    start_fc = _get_frame_count(env) if (env and label) else 0

    for i in range(traj.shape[0]):
        b = base_cmd if lock_base else traj[i, 0:3]
        step_fn(make_action(traj[i, 3:10], gripper, b))

    # Hold final target until arm + base settle
    final_arm = traj[-1, 3:10]
    final_base = base_cmd if lock_base else traj[-1, 0:3]
    final_act = make_action(final_arm, gripper, final_base)
    print(f"    Traj base: {traj[0, 0:3]} -> {traj[-1, 0:3]}"
          f"  (cmd: {final_base})")

    if robot is not None:
        for si in range(settle_steps):
            step_fn(final_act)
            qpos = get_robot_qpos(robot)
            arm_err = np.max(np.abs(qpos[3:10] - final_arm))
            base_err = np.max(np.abs(qpos[0:3] - final_base))
            if arm_err < settle_thresh and base_err < settle_thresh:
                break
        eef_link = next((l for l in robot.get_links()
                         if l.get_name() == 'eef'), None)
        eef_str = f", EEF: {eef_link.pose.p[0].cpu().numpy()}" if eef_link else ""
        print(f"    Settled in {si+1} steps "
              f"(arm_err={arm_err:.4f}, base_err={base_err:.4f}){eef_str}")

    if env and label:
        overlay_text_on_frames(env, label,
                               n_frames=_get_frame_count(env) - start_fc)


def plan_and_move(label, planner, pw, pose, qpos_fn, mask, step_fn,
                  gripper, planning_time=5.0, env=None, robot=None):
    """Plan to a pose and execute. Returns True on success."""
    pmodel = pw.get_planned_articulations()[0].get_pinocchio_model()

    sync_planner(planner)
    cq = qpos_fn() if callable(qpos_fn) else qpos_fn
    result = planner.plan_pose(pose, cq, mask=mask, planning_time=planning_time)

    if result['status'] == 'Success':
        n = result['position'].shape[0]
        print(f"  {label}: OK  ({n} waypoints, {result['duration']:.2f}s)"
              f"  target={np.array(pose.p)}")
        lock_base = (isinstance(mask, np.ndarray) and mask[0])
        execute_trajectory(result['position'], step_fn, gripper,
                           lock_base=lock_base, env=env,
                           label=label, robot=robot)
        check_joint_limits(result['position'][-1],
                           pmodel.get_joint_limits(),
                           pmodel.get_joint_names(),
                           label=f"at {label} endpoint")
        return True

    diagnose_failure(label, result, pose, cq, planner, pw, mask)
    return False


def diagnose_failure(label, result, target_pose, current_qpos,
                     planner, pw, mask=None):
    """Print detailed diagnostics when planning fails."""
    print(f"  {label}: FAILED — {result['status']}")
    print(f"    Target pos:  {np.array(target_pose.p)}")
    print(f"    Target quat: {np.array(target_pose.q)}")
    print(f"    Current qpos (arm): {current_qpos[3:10]}")

    pmodel = pw.get_planned_articulations()[0].get_pinocchio_model()
    check_joint_limits(current_qpos, pmodel.get_joint_limits(),
                       pmodel.get_joint_names(), label="current")

    obj_names = pw.get_object_names()
    print(f"    Obstacles ({len(obj_names)}):")
    for oname in obj_names:
        print(f"      - {oname}  pose={pw.get_object(oname).pose}")

    collisions = pw.check_collision()
    if collisions:
        print(f"    Current-state collisions ({len(collisions)}):")
        for c in collisions:
            print(f"      - {c.link_name1}({c.object_name1}) "
                  f"<-> {c.link_name2}({c.object_name2})")

    try:
        ik_status, ik_solutions = planner.IK(
            target_pose, current_qpos, mask=mask, n_init_qpos=40, verbose=True)
        if ik_solutions is not None:
            print(f"    IK check: {len(ik_solutions)} solution(s) "
                  f"— RRT failed to find collision-free path")
            for i, q in enumerate(ik_solutions):
                print(f"      solution {i}: arm_qpos={q[3:10]}")
        else:
            print(f"    IK check: {ik_status}")
            ik2, sols2 = planner.IK(
                target_pose, current_qpos, mask=None,
                n_init_qpos=40, verbose=True)
            if sols2 is not None:
                print(f"    IK (no mask): {len(sols2)} solution(s) "
                      f"— mask is too restrictive")
            else:
                print(f"    IK (no mask): {ik2} — pose is unreachable")
    except Exception as e:
        print(f"    IK check error: {e}")


def actuate_gripper(step_fn, env, robot, gripper_val, label, n_steps=30):
    """Open or close gripper for n_steps with text overlay."""
    qpos = get_robot_qpos(robot)
    arm, base = qpos[3:10], qpos[0:3]
    action = make_action(arm, gripper_val, base)
    start_fc = _get_frame_count(env)
    for _ in range(n_steps):
        step_fn(action)
    overlay_text_on_frames(env, label,
                           n_frames=_get_frame_count(env) - start_fc)


def build_grasp_poses(block_pos, arm_base):
    """Compute grasp target poses for blocks: Angled45 + Top-Down only."""
    yaw = np.arctan2(block_pos[1] - arm_base[1], block_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return [
        ('Angled45',
         block_pos + [-0.02 * cos_y, -0.02 * sin_y, 0.02],
         list(euler2quat(0, 3 * np.pi / 4, yaw))),
        ('Top-Down',
         block_pos + [0, 0, 0],
         [0, 1, 0, 0]),
    ]


# ─── Camera setup ────────────────────────────────────────────────────────────

def setup_camera():
    """Override default camera to a raised, zoomed-out viewpoint."""
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
    from mani_skill.utils import sapien_utils as ms_sapien_utils

    @property
    def _cam(self):
        pose = ms_sapien_utils.look_at(
            eye=[0.9, 1.0, 1.5], target=[0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
    PickCubeEnv._default_human_render_camera_configs = _cam


# ─── Scene setup ──────────────────────────────────────────────────────────────

def create_table(scene, x, height):
    tb = scene.create_actor_builder()
    half = [0.3, 0.3, height / 2]
    tb.add_box_collision(half_size=half)
    tb.add_box_visual(half_size=half,
                      material=sapien.render.RenderMaterial(
                          base_color=[0.6, 0.4, 0.2, 1.0]))
    table = tb.build_static(name="table")
    table.set_pose(sapien.Pose(p=[x, 0, height / 2]))
    return table


def create_block(scene, x, table_height):
    bb = scene.create_actor_builder()
    half = [0.02, 0.02, 0.02]
    bb.add_box_collision(half_size=half)
    bb.add_box_visual(half_size=half,
                      material=sapien.render.RenderMaterial(
                          base_color=[1.0, 0.2, 0.2, 1.0]))
    block = bb.build(name="red_block")
    block.set_pose(sapien.Pose(p=[x, 0, table_height + 0.04]))
    return block


# ─── Grasp execution ─────────────────────────────────────────────────────────

def execute_single_grasp(gi, name, target_p, target_q, n_grasps,
                         robot, planner, pw, step_fn, env, viz_dir=None):
    """Execute a full pick-and-place cycle for one grasp strategy.

    Steps: IK-seeded pre-grasp → approach → close → lift → drop → return home.
    Returns True if the grasp completed successfully.
    """
    grasp_label = f"[{gi+1}/{n_grasps}] {name}"
    target_q_arr = np.array(target_q)
    print(f"  Base: {get_robot_qpos(robot)[:3]}  Arm: {get_robot_qpos(robot)[3:10]}")

    def _snap(stage_name):
        if viz_dir:
            sync_planner(planner)
            slug = name.lower().replace(' ', '_').replace('-', '_')
            save_planning_world(pw, os.path.join(viz_dir, f"{gi}_{slug}_{stage_name}"))

    def hold_open():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_OPEN, q[:3])

    def hold_closed():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_CLOSED, q[:3])

    # --- Solve IK for grasp, then seed pre-grasp from it ---
    approach_pose = MPPose(p=np.array(target_p), q=target_q_arr)
    pre_pose = MPPose(p=np.array(target_p) + [0, 0, PRE_GRASP_HEIGHT],
                      q=target_q_arr)

    sync_planner(planner)
    cq = get_robot_qpos(robot)

    approach_pose_base = planner._transform_goal_to_wrt_base(approach_pose)
    pre_pose_base = planner._transform_goal_to_wrt_base(pre_pose)

    # 1a. Solve grasp IK (arm-only first, whole-body fallback)
    q_grasp = None
    grasp_mask = None
    for mask_name, mask in [("arm-only", MASK_ARM_ONLY),
                            ("whole-body", MASK_WHOLE_BODY)]:
        status, solutions = planner.IK(
            approach_pose_base, cq, mask=mask, n_init_qpos=40,
            return_closest=True)
        if solutions is not None:
            q_grasp = solutions
            grasp_mask = mask
            print(f"  Grasp IK ({mask_name}): OK (seeding pre-grasp)")
            break
    if q_grasp is None:
        print(f"  Grasp IK: FAILED — {status}")
        return False

    # 1b. Solve pre-grasp IK seeded from grasp solution
    _, pregrasp_sols = planner.IK(
        pre_pose_base, q_grasp, mask=grasp_mask, n_init_qpos=40,
        return_closest=True)
    if pregrasp_sols is None:
        print(f"  Pre-grasp IK: FAILED")
        return False
    print(f"  Pre-grasp IK: OK")

    # 1c. Plan path from current config to pre-grasp config
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    result = planner.plan_qpos([pregrasp_sols], cq, planning_time=5.0)
    if result['status'] != 'Success':
        print(f"  Pre-grasp path: FAILED — {result['status']}")
        return False
    print(f"  Pre-grasp path: OK  ({result['position'].shape[0]} waypoints, "
          f"{result['duration']:.2f}s)  target={np.array(pre_pose.p)}")

    used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
    motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

    execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                       lock_base=used_arm_only, env=env,
                       label="Pre-grasp", robot=robot)
    pause_with_label(env, step_fn, hold_open(), f"{grasp_label} - Pre-grasp")
    _snap("1_pregrasp")

    # 2. Approach
    plan_and_move("Approach", planner, pw, approach_pose,
                  lambda: get_robot_qpos(robot), motion_mask, step_fn,
                  GRIPPER_OPEN, env=env, robot=robot)
    pause_with_label(env, step_fn, hold_open(), f"{grasp_label} - Approach")
    _snap("2_approach")

    # 3. Close gripper
    actuate_gripper(step_fn, env, robot, GRIPPER_CLOSED,
                    f"{grasp_label} - Closing")
    pause_with_label(env, step_fn, hold_closed(), f"{grasp_label} - Grasped")
    _snap("3_grasped")

    # 4. Lift
    lift_pose = MPPose(p=np.array(target_p) + [0, 0, LIFT_HEIGHT],
                       q=target_q_arr)
    plan_and_move("Lift", planner, pw, lift_pose,
                  lambda: get_robot_qpos(robot), motion_mask, step_fn,
                  GRIPPER_CLOSED, planning_time=3.0, env=env, robot=robot)
    pause_with_label(env, step_fn, hold_closed(), f"{grasp_label} - Lifted")
    _snap("4_lifted")

    # 5. Drop block
    actuate_gripper(step_fn, env, robot, GRIPPER_OPEN,
                    f"{grasp_label} - Dropping")
    for _ in range(30):  # let block settle
        step_fn(hold_open())

    # 6. Return to home (plan, no teleport)
    if gi < n_grasps - 1:
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        home_qpos = cq.copy()
        home_qpos[3:10] = ARM_HOME
        home_qpos[10:] = 0.0
        r_home = planner.plan_qpos([home_qpos], cq, planning_time=5.0)
        if r_home['status'] == 'Success':
            print(f"  Return: OK  ({r_home['position'].shape[0]} waypoints)")
            execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                               env=env, label=f"{grasp_label} - Return",
                               robot=robot)
        else:
            print(f"  Return: FAILED — {r_home['status']}, teleporting")
            robot.set_qpos(torch.tensor(
                home_qpos, dtype=torch.float32).unsqueeze(0))
        wait_until_stable(step_fn,
                          make_action(ARM_HOME, GRIPPER_OPEN,
                                      get_robot_qpos(robot)[:3]),
                          robot, max_steps=100)

    return True


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Table-top grasp test")
    parser.add_argument('--render', default='human',
                        choices=['human', 'rgb_array'],
                        help='human=GUI window, rgb_array=save video')
    parser.add_argument('--robot-x', type=float, default=-0.3)
    parser.add_argument('--table-x', type=float, default=0.0)
    parser.add_argument('--table-height', type=float, default=0.762,
                        help='Table height in metres (default: 30 in)')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Save planning-world collision meshes (glb) '
                             'at each grasp stage to this directory')
    args = parser.parse_args()

    # --- Environment ---
    setup_camera()
    env = gym.make('PickCube-v1', num_envs=1, robot_uids='tidyverse',
                   control_mode='whole_body', render_mode=args.render)
    video_dir = os.path.join(os.path.dirname(__file__), 'videos')
    if args.render == 'rgb_array':
        env = RecordEpisode(env, output_dir=video_dir, save_video=True,
                            max_steps_per_video=10000, video_fps=30)
    env.reset(seed=0)
    robot = env.unwrapped.agent.robot
    scene_ms = env.unwrapped.scene
    is_human = (args.render == 'human')

    def step_fn(action):
        env.step(action)
        if is_human:
            env.render()

    # --- Scene objects ---
    robot.set_pose(sapien.Pose(p=[args.robot_x, 0, 0]))
    create_table(scene_ms, args.table_x, args.table_height)
    block = create_block(scene_ms, args.table_x, args.table_height)

    # --- Stabilize ---
    base_cmd = np.array([args.robot_x, 0.0, 0.0])
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    print("Waiting for robot to stabilize...")
    wait_until_stable(step_fn, hold, robot)

    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    block_pos = block.pose.p[0].cpu().numpy()
    print(f"Arm base: {arm_base}")
    print(f"Block:    {block_pos}  "
          f"(dist XY: {np.linalg.norm(arm_base[:2] - block_pos[:2]):.3f}m)")

    # --- Planner ---
    pw = SapienPlanningWorld(scene_ms.sub_scenes[0], [robot._objs[0]])
    eef = next(n for n in pw.get_planned_articulations()[0]
               .get_pinocchio_model().get_link_names() if 'eef' in n)
    planner = SapienPlanner(pw, move_group=eef)

    # --- Snapshot initial planning world ---
    if args.viz_dir:
        sync_planner(planner)
        save_planning_world(pw, os.path.join(args.viz_dir, "initial_home"))

    # --- Grasp loop ---
    grasps = build_grasp_poses(block_pos, arm_base)
    for gi, (name, target_p, target_q) in enumerate(grasps):
        print(f"\n{'='*50}")
        print(f"[{gi + 1}/{len(grasps)}] {name}")
        print(f"  Target: pos={target_p}  quat={target_q}")
        execute_single_grasp(gi, name, target_p, target_q, len(grasps),
                             robot, planner, pw, step_fn, env,
                             viz_dir=args.viz_dir)

    # --- Finish ---
    if is_human:
        print("\nDone! Close the window to exit.")
        while True:
            q = get_robot_qpos(robot)
            env.step(make_action(q[3:10], GRIPPER_OPEN, q[:3]))
            env.render()
    else:
        env.close()
        print(f"\nDone! Video saved to {video_dir}/")


if __name__ == '__main__':
    main()
