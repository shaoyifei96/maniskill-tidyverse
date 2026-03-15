"""Table-top grasp test: table + red block, multi-angle pick with diagnostics."""
import sys
import os
import argparse
import numpy as np
import torch
import cv2
import sapien
import gymnasium as gym

sys.path.insert(0, os.path.dirname(__file__))
import tidyverse_agent  # noqa: F401 — registers 'tidyverse' robot
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

def overlay_text_on_frames(env, text, n_frames=None):
    """Burn text onto the last n_frames of the recorded video.

    Works with RecordEpisode wrapper (rgb_array mode).
    If n_frames is None, overlays on all frames.
    """
    recorder = env
    while hasattr(recorder, 'env'):
        if hasattr(recorder, 'render_images'):
            break
        recorder = recorder.env
    if not hasattr(recorder, 'render_images') or not recorder.render_images:
        return
    frames = recorder.render_images
    start = max(0, len(frames) - n_frames) if n_frames else 0
    for i in range(start, len(frames)):
        img = frames[i]
        if isinstance(img, np.ndarray):
            img = img.copy()
            h, w = img.shape[:2]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.5, h / 600)
            thickness = max(1, int(h / 300))
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x, y = 10, th + 10
            # Background rectangle
            cv2.rectangle(img, (x - 5, y - th - 5),
                          (x + tw + 5, y + 5), (0, 0, 0), -1)
            cv2.putText(img, text, (x, y), font, scale,
                        (255, 255, 255), thickness, cv2.LINE_AA)
            frames[i] = img


def pause_with_label(env, step_fn, hold_action, label, fps=30, seconds=1.0):
    """Hold current pose for `seconds`, overlaying label text on frames."""
    n_steps = int(fps * seconds)
    start_frame_count = _get_frame_count(env)
    for _ in range(n_steps):
        step_fn(hold_action)
    added = _get_frame_count(env) - start_frame_count
    overlay_text_on_frames(env, label, n_frames=added)


def _get_frame_count(env):
    """Get current number of recorded frames."""
    recorder = env
    while hasattr(recorder, 'env'):
        if hasattr(recorder, 'render_images'):
            return len(recorder.render_images)
        recorder = recorder.env
    return 0


def make_action(arm_qpos, gripper, base_cmd):
    """Build a single action tensor: [arm(7), gripper(1), base(3)]."""
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


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
    """Print warnings for any joints at or beyond their limits."""
    qi = 0  # index into flat qpos
    for limits, name in zip(joint_limits, joint_names):
        # limits shape: (ndof_for_joint, 2) — columns are [lo, hi]
        if limits.ndim == 2:
            for d in range(limits.shape[0]):
                if qi >= len(qpos):
                    return
                lo, hi = limits[d, 0], limits[d, 1]
                val = qpos[qi]
                margin = (hi - lo) * 0.02  # 2% of range
                if val <= lo + margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {val:.4f} "
                          f"at lower limit {lo:.4f}")
                elif val >= hi - margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {val:.4f} "
                          f"at upper limit {hi:.4f}")
                qi += 1
        elif limits.ndim == 1 and limits.shape[0] >= 2:
            if qi >= len(qpos):
                return
            lo, hi = limits[0], limits[1]
            val = qpos[qi]
            margin = (hi - lo) * 0.02
            if val <= lo + margin:
                print(f"    JOINT LIMIT {label}: {name} = {val:.4f} "
                      f"at lower limit {lo:.4f}")
            elif val >= hi - margin:
                print(f"    JOINT LIMIT {label}: {name} = {val:.4f} "
                      f"at upper limit {hi:.4f}")
            qi += 1
        else:
            # Unknown format, skip
            qi += 1


def execute_trajectory(traj, step_fn, gripper, lock_base=False,
                       env=None, label=None, robot=None,
                       settle_thresh=0.01, settle_steps=100):
    """Execute a planned trajectory, sending arm + base targets each step.

    If lock_base is True, use the base values from the first waypoint
    for all steps (RRT may drift base even with arm-only IK mask).
    If env and label are provided, overlay label text on recorded frames.
    After the trajectory, holds the final target until the arm converges.
    """
    base_cmd = traj[0, 0:3] if lock_base else None
    start_fc = _get_frame_count(env) if (env and label) else 0
    for i in range(traj.shape[0]):
        b = base_cmd if lock_base else traj[i, 0:3]
        step_fn(make_action(traj[i, 3:10], gripper, b))
    # Hold final target until arm settles
    final_arm = traj[-1, 3:10]
    final_base = base_cmd if lock_base else traj[-1, 0:3]
    final_act = make_action(final_arm, gripper, final_base)
    print(f"    Traj base: {traj[0, 0:3]} -> {traj[-1, 0:3]}"
          f"  (cmd: {final_base})")
    if robot is not None:
        final_full = np.concatenate([final_base, final_arm])
        for si in range(settle_steps):
            step_fn(final_act)
            qpos = robot.get_qpos().cpu().numpy()[0]
            # Check both base and arm convergence
            arm_err = np.max(np.abs(qpos[3:10] - final_arm))
            base_err = np.max(np.abs(qpos[0:3] - final_base))
            if arm_err < settle_thresh and base_err < settle_thresh:
                break
        # Report EEF position
        eef_link = next((l for l in robot.get_links()
                         if l.get_name() == 'eef'), None)
        if eef_link is not None:
            eef_pos = eef_link.pose.p[0].cpu().numpy()
            print(f"    Settled in {si+1} steps (arm_err={arm_err:.4f}, "
                  f"base_err={base_err:.4f}), EEF: {eef_pos}")
        else:
            print(f"    Settled in {si+1} steps (arm_err={arm_err:.4f}, "
                  f"base_err={base_err:.4f})")
    if env and label:
        added = _get_frame_count(env) - start_fc
        overlay_text_on_frames(env, label, n_frames=added)


def plan_and_move(label, planner, pw, pose, qpos, masks, step_fn,
                  gripper, planning_time=5.0, env=None, robot=None):
    """Plan to a pose and execute, trying each mask in order.

    Args:
        masks: single mask array, or list of (name, mask) tuples for fallback.
    Returns True on success.
    """
    pmodel = pw.get_planned_articulations()[0].get_pinocchio_model()
    joint_limits = pmodel.get_joint_limits()
    joint_names = pmodel.get_joint_names()

    # Normalize to list of (name, mask) pairs
    if isinstance(masks, np.ndarray):
        attempts = [(None, masks)]
    else:
        attempts = masks
    for mode_name, mask in attempts:
        # Re-sync planner state before each attempt (IK diagnostics may modify it)
        try:
            planner.update_from_simulation()
        except Exception:
            pass
        cq = qpos() if callable(qpos) else qpos
        suffix = f" ({mode_name})" if mode_name else ""
        result = planner.plan_pose(pose, cq, mask=mask, planning_time=planning_time)
        if result['status'] == 'Success':
            n = result['position'].shape[0]
            print(f"  {label}{suffix}: OK  ({n} waypoints, {result['duration']:.2f}s)"
                  f"  target={np.array(pose.p)}")
            # Lock base if base joints are masked (arm-only planning)
            lock_base = (isinstance(mask, np.ndarray) and mask[0])
            display_label = f"{label}{suffix}"
            execute_trajectory(result['position'], step_fn, gripper,
                               lock_base=lock_base, env=env,
                               label=display_label, robot=robot)
            # Check joint limits at trajectory endpoint
            endpoint = result['position'][-1]
            check_joint_limits(endpoint, joint_limits, joint_names,
                               label=f"at {label} endpoint")
            return True
        diagnose_failure(f"{label}{suffix}", result, pose, cq, planner, pw, mask)
    return False


def diagnose_failure(label, result, target_pose, current_qpos,
                     planner, pw, mask=None):
    """Print detailed diagnostics when planning fails."""
    print(f"  {label}: FAILED — {result['status']}")
    print(f"    Target pos:  {np.array(target_pose.p)}")
    print(f"    Target quat: {np.array(target_pose.q)}")
    print(f"    Current qpos (arm): {current_qpos[3:10]}")

    # Joint limits check on current state
    pmodel = pw.get_planned_articulations()[0].get_pinocchio_model()
    check_joint_limits(current_qpos, pmodel.get_joint_limits(),
                       pmodel.get_joint_names(), label="current")

    # Obstacles
    obj_names = pw.get_object_names()
    print(f"    Obstacles ({len(obj_names)}):")
    for oname in obj_names:
        obj = pw.get_object(oname)
        print(f"      - {oname}  pose={obj.pose}")

    # Current-state collisions
    collisions = pw.check_collision()
    if collisions:
        print(f"    Current-state collisions ({len(collisions)}):")
        for c in collisions:
            print(f"      - {c.link_name1}({c.object_name1}) "
                  f"<-> {c.link_name2}({c.object_name2})")

    # Standalone IK check
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
            # Retry without mask to isolate the cause
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


def build_grasp_poses(block_pos, arm_base):
    """Compute grasp target poses relative to the block."""
    yaw = np.arctan2(block_pos[1] - arm_base[1], block_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    return [
        ('Top-Down',
         block_pos + [0, 0, 0],
         [0, 1, 0, 0]),
        ('Front',
         block_pos + [-0.06 * cos_y, -0.06 * sin_y, 0.08],
         list(R.from_euler('yz', [np.pi / 2, yaw]).as_quat()[[3, 0, 1, 2]])),
        ('Angled45',
         block_pos + [-0.02 * cos_y, -0.02 * sin_y, 0.02],
         list(euler2quat(0, 3 * np.pi / 4, yaw))),
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
    scene = env.unwrapped.scene.sub_scenes[0]
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
    print(f"Block:    {block_pos}  (dist XY: {np.linalg.norm(arm_base[:2] - block_pos[:2]):.3f}m)")

    # --- Planner ---
    pw = SapienPlanningWorld(scene, [robot._objs[0]])
    eef = next(n for n in pw.get_planned_articulations()[0]
               .get_pinocchio_model().get_link_names() if 'eef' in n)
    planner = SapienPlanner(pw, move_group=eef)

    # --- Grasp loop ---
    grasps = build_grasp_poses(block_pos, arm_base)
    get_qpos = lambda: robot.get_qpos().cpu().numpy()[0]

    for gi, (name, target_p, target_q) in enumerate(grasps):
        print(f"\n{'='*50}")
        print(f"[{gi + 1}/{len(grasps)}] {name}")
        print(f"  Target: pos={target_p}  quat={target_q}")

        print(f"  Base: {get_qpos()[:3]}  Arm: {get_qpos()[3:10]}")

        target_q_arr = np.array(target_q)
        # Pre-grasp: try arm-only first, fall back to whole-body
        pregrasp_masks = [("arm-only", MASK_ARM_ONLY),
                          ("whole-body", MASK_WHOLE_BODY)]

        grasp_label = f"[{gi+1}/{len(grasps)}] {name}"
        get_base = lambda: get_qpos()[:3]
        hold_act = lambda: make_action(get_qpos()[3:10], GRIPPER_OPEN, get_base())
        hold_closed = lambda: make_action(get_qpos()[3:10], GRIPPER_CLOSED, get_base())

        # Solve grasp IK first, then seed pre-grasp IK from it
        # so the pre-grasp arm config is close to what approach needs.
        approach_pose = MPPose(p=np.array(target_p), q=target_q_arr)
        pre_pose = MPPose(p=np.array(target_p) + [0, 0, PRE_GRASP_HEIGHT],
                          q=target_q_arr)

        try:
            planner.update_from_simulation()
        except Exception:
            pass
        cq = get_qpos()

        # Transform poses to planner base frame (IK expects base-frame poses)
        approach_pose_base = planner._transform_goal_to_wrt_base(approach_pose)
        pre_pose_base = planner._transform_goal_to_wrt_base(pre_pose)

        # 1a. Solve grasp IK (don't execute, just get target config)
        for _, mask in pregrasp_masks:
            grasp_ik_status, grasp_ik_solutions = planner.IK(
                approach_pose_base, cq, mask=mask, n_init_qpos=40,
                return_closest=True)
            if grasp_ik_solutions is not None:
                q_grasp = grasp_ik_solutions
                print(f"  Grasp IK: OK (using as seed for pre-grasp)")
                break
        else:
            print(f"  Grasp IK: FAILED — {grasp_ik_status}")
            print("  SKIPPED — no grasp IK solution")
            continue

        # 1b. Solve pre-grasp IK seeded from grasp solution
        pregrasp_ik_status, pregrasp_ik_solutions = planner.IK(
            pre_pose_base, q_grasp, mask=mask, n_init_qpos=40,
            return_closest=True)
        if pregrasp_ik_solutions is None:
            print(f"  Pre-grasp IK: FAILED — {pregrasp_ik_status}")
            print("  SKIPPED — no pre-grasp IK solution")
            continue
        q_pregrasp = pregrasp_ik_solutions
        print(f"  Pre-grasp IK: OK")

        # 1c. Plan path from current config to pre-grasp config
        try:
            planner.update_from_simulation()
        except Exception:
            pass
        cq = get_qpos()
        result = planner.plan_qpos([q_pregrasp], cq, planning_time=5.0)
        if result['status'] != 'Success':
            print(f"  Pre-grasp path: FAILED — {result['status']}")
            print("  SKIPPED — no path to pre-grasp")
            continue
        n = result['position'].shape[0]
        print(f"  Pre-grasp path: OK  ({n} waypoints, {result['duration']:.2f}s)"
              f"  target={np.array(pre_pose.p)}")
        # Use the mask that succeeded for grasp IK
        used_arm_only = (isinstance(mask, np.ndarray) and mask[0])
        lock_base = used_arm_only
        # If grasp IK needed whole-body, approach/lift may too
        motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                           lock_base=lock_base, env=env,
                           label=f"Pre-grasp", robot=robot)
        pause_with_label(env, step_fn, hold_act(), f"{grasp_label} - Pre-grasp")

        # 2. Approach
        plan_and_move("Approach", planner, pw, approach_pose,
                      get_qpos, motion_mask, step_fn,
                      GRIPPER_OPEN, env=env, robot=robot)
        pause_with_label(env, step_fn, hold_act(), f"{grasp_label} - Approach")

        # 3. Close gripper
        aq = get_qpos()[3:10]
        cur_base = get_base()
        start_fc = _get_frame_count(env)
        for _ in range(30):
            step_fn(make_action(aq, GRIPPER_CLOSED, cur_base))
        overlay_text_on_frames(env, f"{grasp_label} - Closing",
                               n_frames=_get_frame_count(env) - start_fc)
        pause_with_label(env, step_fn, hold_closed(),
                         f"{grasp_label} - Grasped")

        # 4. Lift
        lift_pose = MPPose(p=np.array(target_p) + [0, 0, LIFT_HEIGHT],
                           q=target_q_arr)
        plan_and_move("Lift", planner, pw, lift_pose,
                      get_qpos, motion_mask, step_fn,
                      GRIPPER_CLOSED, planning_time=3.0, env=env,
                      robot=robot)
        pause_with_label(env, step_fn, hold_closed(),
                         f"{grasp_label} - Lifted")

        # 5. Open gripper to drop block
        aq = get_qpos()[3:10]
        cur_base = get_base()
        start_fc = _get_frame_count(env)
        for _ in range(30):
            step_fn(make_action(aq, GRIPPER_OPEN, cur_base))
        overlay_text_on_frames(env, f"{grasp_label} - Dropping",
                               n_frames=_get_frame_count(env) - start_fc)
        # Let block settle
        for _ in range(30):
            step_fn(make_action(aq, GRIPPER_OPEN, cur_base))

        # 6. Return to home (plan, no teleport)
        if gi < len(grasps) - 1:
            try:
                planner.update_from_simulation()
            except Exception:
                pass
            cq = get_qpos()
            home_qpos = cq.copy()
            home_qpos[3:10] = ARM_HOME
            home_qpos[10:] = 0.0
            r_home = planner.plan_qpos(
                [home_qpos], cq, planning_time=5.0)
            if r_home['status'] == 'Success':
                print(f"  Return: OK  ({r_home['position'].shape[0]} waypoints)")
                execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                                   env=env, label=f"{grasp_label} - Return",
                                   robot=robot)
            else:
                print(f"  Return: FAILED — {r_home['status']}, teleporting")
                robot.set_qpos(torch.tensor(
                    home_qpos, dtype=torch.float32).unsqueeze(0))
            cur_hold = make_action(ARM_HOME, GRIPPER_OPEN, get_qpos()[:3])
            wait_until_stable(step_fn, cur_hold, robot, max_steps=100)

    # --- Finish ---
    if is_human:
        print("\nDone! Close the window to exit.")
        while True:
            cur_hold = make_action(get_qpos()[3:10], GRIPPER_OPEN, get_qpos()[:3])
            env.step(cur_hold)
            env.render()
    else:
        env.close()
        print(f"\nDone! Video saved to {video_dir}/")


if __name__ == '__main__':
    main()
