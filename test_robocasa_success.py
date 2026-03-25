#!/usr/bin/env python3
"""RoboCasa kitchen grasp test WITH per-step success flags.

Based on test_robocasa_grasp.py — same grasp pipeline (pre-grasp → approach →
close → lift → drop → home) but instruments each phase with success condition
checks ported from RoboCasa (see success_utils.py).

Usage:
    # GUI — watch the robot attempt grasps with flag output
    python test_robocasa_success.py --render human --seed 0

    # Headless — save video + JSON flags
    python test_robocasa_success.py --render rgb_array --seed 0

    # Limit to N nearest cubes
    python test_robocasa_success.py --render human --max-cubes 5
"""
import sys, os, signal, argparse, json, time, warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np
import torch, sapien, cv2
import gymnasium as gym

import maniskill_tidyverse.tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs

from maniskill_tidyverse.success_utils import compute_step_flags, format_flags
from maniskill_tidyverse.task_registry import (get_grasp_instruction, get_pick_place_instruction,
                           check_phase_expectations, EXPECTED_FLAGS)

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
from transforms3d.euler import euler2mat, euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv


# ─── Video writer ─────────────────────────────────────────────────────────────

class VideoWriter:
    """Accumulate frames and write mp4 on close."""
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def add_frame(self, frame):
        if frame.ndim == 4:
            frame = frame[0]
        h, w = frame.shape[:2]
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer:
            self.writer.release()
            print(f"Video saved: {self.path} ({self.frame_count} frames)")


# ─── Collision logger ─────────────────────────────────────────────────────────

class CollisionLogger:
    def __init__(self, robot, scene, env, img_dir, render_mode='human'):
        self.robot = robot
        self.scene = scene
        self.env = env
        self.img_dir = img_dir
        self.render_mode = render_mode
        import shutil
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        self.robot_entity_names = set()
        for link in robot.get_links():
            self.robot_entity_names.add(link.get_name())
        self.seen_pairs = set()
        self.collision_count = 0
        self.step_count = 0

    def check(self, step_label=""):
        self.step_count += 1
        try:
            contacts = self.scene.get_contacts()
        except Exception:
            return
        for contact in contacts:
            if not contact.points:
                continue
            impulse = np.sum([pt.impulse for pt in contact.points], axis=0)
            if np.linalg.norm(impulse) < 1e-4:
                continue
            b0, b1 = contact.bodies[0], contact.bodies[1]
            name0 = b0.entity.name if b0.entity else str(b0)
            name1 = b1.entity.name if b1.entity else str(b1)
            is_robot0 = name0 in self.robot_entity_names
            is_robot1 = name1 in self.robot_entity_names
            if not (is_robot0 or is_robot1):
                continue
            if is_robot0 and is_robot1:
                continue
            pair = frozenset((name0, name1))
            if pair not in self.seen_pairs:
                self.seen_pairs.add(pair)
                self.collision_count += 1
                robot_part = name0 if is_robot0 else name1
                other_part = name1 if is_robot0 else name0
                sep = min(pt.separation for pt in contact.points)
                imp_mag = np.linalg.norm(impulse)
                print(f"  COLLISION #{self.collision_count} step={self.step_count}: "
                      f"{robot_part} <-> {other_part}  "
                      f"impulse={imp_mag:.4f}  sep={sep:.4f}  "
                      f"[{step_label}]")

    def summary(self):
        print(f"\nCollision summary: {self.collision_count} unique collision pairs "
              f"detected over {self.step_count} steps")
        for pair in sorted(self.seen_pairs, key=lambda p: sorted(p)):
            names = sorted(pair)
            print(f"  - {names[0]} <-> {names[1]}")


from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    SingleCabinet, HingeCabinet, OpenCabinet, Drawer,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.accessories import CoffeeMachine
from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Floor, Wall
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


# ─── Constants ─────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.15
CUBE_HALF = 0.02

MASK_ARM_ONLY  = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)

PLANNING_TIMEOUT = 15
IK_TIMEOUT = 8

COLORS = [
    [1.0, 0.0, 0.0, 1], [0.0, 0.8, 0.0, 1], [0.0, 0.3, 1.0, 1],
    [1.0, 0.7, 0.0, 1], [0.8, 0.0, 0.8, 1], [0.0, 0.8, 0.8, 1],
    [1.0, 1.0, 0.0, 1], [1.0, 0.4, 0.7, 1], [0.6, 0.3, 0.0, 1],
    [0.5, 0.5, 0.5, 1],
]


# ─── Timeout handler ──────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("planning timeout")

signal.signal(signal.SIGALRM, _timeout_handler)


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


# ─── Placement helpers ─────────────────────────────────────────────────────────

def local_to_world(fixture, offset):
    rot_mat = euler2mat(0, 0, fixture.rot)
    return np.array(fixture.pos) + rot_mat @ np.array(offset)


def spawn_cube(scene, name, pos, color):
    builder = scene.create_actor_builder()
    hs = np.array([CUBE_HALF] * 3)
    builder.add_box_collision(half_size=hs)
    builder.add_box_visual(
        half_size=hs,
        material=sapien.render.RenderMaterial(base_color=color),
    )
    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(p=pos))
    return actor


def _region_placements(fix, regions):
    results = []
    for rname, region in regions.items():
        offset = np.array(region["offset"], dtype=float)
        wp = local_to_world(fix, offset)
        wp[2] = fix.pos[2] + offset[2]
        results.append((rname, wp))
    return results


def _int_sites_placement(fix, suffix="interior"):
    try:
        p0, px, py, pz = fix.get_int_sites()
        cx = np.mean([p0[0], px[0]])
        cy = np.mean([p0[1], py[1]])
        cz = p0[2]
        wp = local_to_world(fix, [cx, cy, cz])
        wp[2] = fix.pos[2] + cz
        return [(suffix, wp)]
    except Exception:
        return []


def collect_placements(fixtures):
    """Enumerate all placement surfaces.

    Returns [(label, world_pos, fixture_type_str, fixture_obj)].
    """
    all_placements = []

    for fname, fix in fixtures.items():
        if isinstance(fix, (Floor, Wall)):
            continue

        positions = []
        ftype = type(fix).__name__

        if isinstance(fix, Counter):
            try:
                regions = fix.get_reset_regions(
                    None, fixtures, ref=None, loc="any", top_size=(0.01, 0.01))
                positions = _region_placements(fix, regions)
            except Exception:
                sz = fix.pos[2] + fix.size[2] / 2
                positions = [("top", np.array([fix.pos[0], fix.pos[1], sz]))]

        elif isinstance(fix, (Stove, Stovetop)):
            try:
                positions = _region_placements(fix, fix.get_reset_regions(None))
            except Exception:
                pass

        elif isinstance(fix, Drawer):
            if 0.4 <= fix.pos[2] + fix.size[2] / 2 <= 1.2:
                positions = _int_sites_placement(fix)

        elif isinstance(fix, (SingleCabinet, HingeCabinet, OpenCabinet)):
            positions = _int_sites_placement(fix)
            top_z = fix.pos[2] + fix.size[2] / 2
            if 1.0 <= top_z <= 1.6:
                positions.append(("top", np.array([fix.pos[0], fix.pos[1], top_z])))

        elif isinstance(fix, Microwave):
            positions = _int_sites_placement(fix)

        elif isinstance(fix, CoffeeMachine):
            try:
                positions = _region_placements(fix, fix.get_reset_regions())
            except Exception:
                pass

        elif isinstance(fix, Sink):
            positions = _int_sites_placement(fix, suffix="basin")

        for rname, pos in positions:
            label = f"{fname}_{rname}"
            all_placements.append((label, pos, ftype, fix))

    return all_placements


# ─── Motion planning helpers ──────────────────────────────────────────────────

def make_action(arm_qpos, gripper, base_cmd):
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


def sync_planner(planner):
    try:
        planner.update_from_simulation()
    except Exception:
        pass


def get_robot_qpos(robot):
    return robot.get_qpos().cpu().numpy()[0]


def wait_until_stable(step_fn, hold, robot, max_steps=300,
                      vel_thresh=1e-3, window=10):
    stable_count = 0
    for si in range(max_steps):
        step_fn(hold)
        qvel = robot.get_qvel().cpu().numpy()[0]
        if np.max(np.abs(qvel)) < vel_thresh:
            stable_count += 1
            if stable_count >= window:
                return si + 1
        else:
            stable_count = 0
    return max_steps


def _find_recorder(env):
    rec = env
    while hasattr(rec, 'env'):
        if hasattr(rec, 'render_images'):
            return rec
        rec = rec.env
    return rec if hasattr(rec, 'render_images') else None


def _get_frame_count(env):
    rec = _find_recorder(env)
    return len(rec.render_images) if rec else 0


def overlay_text_on_frames(env, text, n_frames=None):
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
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thickness)
        x, y = 10, th_ + 10
        cv2.rectangle(img, (x - 5, y - th_ - 5), (x + tw + 5, y + 5), (0, 0, 0), -1)
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        frames[i] = img


def pause_with_label(env, step_fn, hold_action, label, fps=30, seconds=1.0):
    n_steps = int(fps * seconds)
    start_fc = _get_frame_count(env)
    for _ in range(n_steps):
        step_fn(hold_action)
    overlay_text_on_frames(env, label, n_frames=_get_frame_count(env) - start_fc)


def execute_trajectory(traj, step_fn, gripper, lock_base=False,
                       env=None, label=None, robot=None,
                       settle_thresh=0.01, settle_steps=100):
    base_cmd = traj[0, 0:3] if lock_base else None
    start_fc = _get_frame_count(env) if (env and label) else 0

    for i in range(traj.shape[0]):
        b = base_cmd if lock_base else traj[i, 0:3]
        step_fn(make_action(traj[i, 3:10], gripper, b))

    final_arm = traj[-1, 3:10]
    final_base = base_cmd if lock_base else traj[-1, 0:3]
    final_act = make_action(final_arm, gripper, final_base)

    if robot is not None:
        for si in range(settle_steps):
            step_fn(final_act)
            qpos = get_robot_qpos(robot)
            arm_err = np.max(np.abs(qpos[3:10] - final_arm))
            base_err = np.max(np.abs(qpos[0:3] - final_base))
            if arm_err < settle_thresh and base_err < settle_thresh:
                break

    if env and label:
        overlay_text_on_frames(env, label,
                               n_frames=_get_frame_count(env) - start_fc)


def actuate_gripper(step_fn, env, robot, gripper_val, label, n_steps=30):
    qpos = get_robot_qpos(robot)
    arm, base = qpos[3:10], qpos[0:3]
    action = make_action(arm, gripper_val, base)
    start_fc = _get_frame_count(env)
    for _ in range(n_steps):
        step_fn(action)
    overlay_text_on_frames(env, label,
                           n_frames=_get_frame_count(env) - start_fc)


# ─── ACM builder ──────────────────────────────────────────────────────────────

def _get_object_position(pw, name):
    try:
        return np.array(pw.get_object(name).pose.p)
    except Exception:
        return None


def _estimate_art_position(pw, art_name):
    """Estimate articulation world position from its first link's FCL pose."""
    try:
        art = pw.get_articulation(art_name)
        links = art.get_pinocchio_model().get_link_names()
        if links:
            # Use the articulation's base pose
            return np.array(art.get_pinocchio_model().get_random_configuration()[:3])
    except Exception:
        pass
    return None


def build_kitchen_acm(pw, planner, cube_names, mode='strict',
                      robot_pos=None, near_radius=1.5):
    acm = pw.get_allowed_collision_matrix()
    art_names = pw.get_articulation_names()
    robot_link_names = planner.pinocchio_model.get_link_names()
    robot_art = next(n for n in art_names if 'tidyverse' in n.lower())

    print(f"\n  Planning world contents (ACM mode={mode}):")
    print(f"    Robot links ({len(robot_link_names)}): {robot_link_names[:5]}...")
    print(f"    Articulations ({len(art_names)}):")

    relaxed_arts = []
    checked_arts = []
    for an in art_names:
        if an == robot_art:
            print(f"      [ROBOT] {an}")
            continue
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        if mode == 'relaxed':
            relaxed_arts.append(an)
            print(f"      [RELAXED] {an} ({len(fl)} links)")
        else:
            # In strict mode, relax only far-away articulations
            relaxed_arts.append(an)  # default: relax (will override below)
            print(f"      [RELAXED-art] {an} ({len(fl)} links)")

    obj_names = pw.get_object_names()
    checked_objs = [n for n in obj_names if n in cube_names]

    relaxed_static = []
    checked_static = []
    for on in obj_names:
        if on in cube_names:
            continue
        if mode == 'relaxed':
            relaxed_static.append(on)
        else:
            pos = _get_object_position(pw, on)
            if pos is not None and robot_pos is not None:
                dist = np.linalg.norm(pos[:2] - robot_pos[:2])
                if dist > near_radius:
                    relaxed_static.append(on)
                else:
                    checked_static.append((on, dist))
            else:
                checked_static.append((on, -1))

    print(f"    Static objects ({len(obj_names)} total):")
    print(f"      Cubes (always checked): {len(checked_objs)}")
    if mode == 'strict':
        print(f"      Collision-checked (near, <{near_radius}m): {len(checked_static)}")
        for on, d in sorted(checked_static, key=lambda x: x[1]):
            print(f"        - {on}  dist={d:.2f}m")
    print(f"      ACM-relaxed (far): {len(relaxed_static)}")

    # Apply ACM — relax articulated fixtures (mesh too complex for planner)
    # but keep nearby static objects as obstacles
    for an in relaxed_arts:
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        for rl in robot_link_names:
            for f in fl:
                acm.set_entry(rl, f, True)
    for on in relaxed_static:
        for rl in robot_link_names:
            acm.set_entry(rl, on, True)

    # Resolve initial collisions: if the robot starts in collision with a
    # checked static object, we must relax it or the planner can't plan at all.
    sync_planner(planner)
    collisions = pw.check_collision()
    init_relaxed = []
    if collisions:
        print(f"\n  Initial planner collisions ({len(collisions)}):")
        for c in collisions:
            print(f"    {c.link_name1}({c.object_name1}) <-> "
                  f"{c.link_name2}({c.object_name2})")
            # Relax the non-robot object in the collision pair
            for obj_name in [c.object_name1, c.object_name2]:
                if obj_name and obj_name not in cube_names:
                    is_robot_obj = any(obj_name == robot_art or
                                       'tidyverse' in obj_name.lower()
                                       for _ in [1])
                    if not is_robot_obj:
                        for rl in robot_link_names:
                            acm.set_entry(rl, obj_name, True)
                        init_relaxed.append(obj_name)
        if init_relaxed:
            print(f"    Auto-relaxed {len(set(init_relaxed))} objects to resolve initial collisions")
            # Re-check
            sync_planner(planner)
            remaining = pw.check_collision()
            if remaining:
                print(f"    Still {len(remaining)} collisions after auto-relax")
    else:
        print(f"\n  No initial planner collisions")

    n_checked = len(checked_static) - len(set(init_relaxed))
    print(f"\n  ACM summary: {len(relaxed_arts)} fixture articulations relaxed, "
          f"{len(relaxed_static)} far static objects relaxed, "
          f"{max(0, n_checked)} near static objects collision-checked, "
          f"{len(checked_objs)} cubes collision-checked")


# ─── Grasp pose (45-degree only) ──────────────────────────────────────────────

def build_grasp_pose(obj_pos, arm_base):
    """Single 45-degree angled grasp pose aimed from arm toward object."""
    yaw = np.arctan2(obj_pos[1] - arm_base[1], obj_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    p = obj_pos + np.array([-0.02 * cos_y, -0.02 * sin_y, 0.02])
    q = np.array(euler2quat(0, 3 * np.pi / 4, yaw))
    return p, q


def build_place_pose(dest_pos, arm_base):
    """Place pose: top-down above destination."""
    p = dest_pos.copy()
    p[2] += DROP_HEIGHT
    q = np.array([0, 1, 0, 0])  # top-down
    return p, q


DROP_HEIGHT = 0.10  # hover above destination before opening gripper
LIFT_SMALL = 0.01  # lift just 1cm after grasp


# ─── Single pick-and-place attempt ───────────────────────────────────────────

def attempt_pick_place(task_idx, cube_pos, src_label, src_ftype,
                       dest_pos, dest_label, dest_ftype,
                       robot, planner, pw, step_fn, env, total,
                       scene, agent, cube_actor, src_fixture, dest_fixture):
    """Pick from source, place at destination.

    Pipeline: approach → close gripper → lift 1cm → move to dest → open → home.
    """
    original_pos = cube_pos.copy()
    phase_flags = []
    lang = get_pick_place_instruction(src_ftype, dest_ftype, obj_name="cube")
    print(f"  Task: {lang}")

    def _record_flags(phase_name, check_fixture=None):
        fix = check_fixture if check_fixture is not None else src_fixture
        flags = compute_step_flags(scene, agent, cube_actor, fix,
                                   original_pos, phase_name)
        phase_flags.append(flags)
        print(f"    Flags[{phase_name}]: {format_flags(flags)}")

    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()

    tag = f"[{task_idx+1}/{total}] {src_label} -> {dest_label}"

    def hold_open():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_OPEN, q[:3])

    def hold_closed():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_CLOSED, q[:3])

    # ── 1. Approach grasp pose ──
    grasp_p, grasp_q = build_grasp_pose(cube_pos, arm_base)
    grasp_pose = MPPose(p=grasp_p, q=grasp_q)

    sync_planner(planner)
    cq = get_robot_qpos(robot)

    q_grasp = None
    grasp_mask = None
    for mask_name, mask in [("arm-only", MASK_ARM_ONLY),
                            ("whole-body", MASK_WHOLE_BODY)]:
        signal.alarm(IK_TIMEOUT)
        try:
            status, solutions = planner.IK(
                planner._transform_goal_to_wrt_base(grasp_pose),
                cq, mask=mask, n_init_qpos=40, return_closest=True)
        except TimeoutError:
            continue
        finally:
            signal.alarm(0)
        if solutions is not None:
            q_grasp = solutions
            grasp_mask = mask
            print(f"    Grasp IK ({mask_name}): OK")
            break

    if q_grasp is None:
        print(f"    Grasp IK: FAILED")
        return {'outcome': 'unreachable', 'phases': phase_flags, 'lang': lang}

    used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
    motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

    sync_planner(planner)
    cq = get_robot_qpos(robot)
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_app = planner.plan_pose(grasp_pose, cq, mask=motion_mask,
                                   planning_time=5.0)
    except TimeoutError:
        r_app = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)
    if r_app['status'] != 'Success':
        print(f"    Approach: FAILED — {r_app['status']}")
        return {'outcome': 'unreachable', 'phases': phase_flags, 'lang': lang}
    print(f"    Approach: OK ({r_app['position'].shape[0]} wp)")
    execute_trajectory(r_app['position'], step_fn, GRIPPER_OPEN,
                       lock_base=used_arm_only, env=env,
                       label="Approach", robot=robot)
    _record_flags('approach')

    # ── 2. Close gripper ──
    actuate_gripper(step_fn, env, robot, GRIPPER_CLOSED, f"{tag} - Grasp")
    _record_flags('grasp')

    # ── 3. Lift 1cm ──
    lift_pose = MPPose(p=grasp_p + [0, 0, LIFT_SMALL], q=grasp_q)
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_lift = planner.plan_pose(lift_pose, cq, mask=motion_mask,
                                    planning_time=5.0)
    except TimeoutError:
        r_lift = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)
    if r_lift['status'] == 'Success':
        print(f"    Lift: OK ({r_lift['position'].shape[0]} wp)")
        execute_trajectory(r_lift['position'], step_fn, GRIPPER_CLOSED,
                           lock_base=used_arm_only, env=env,
                           label="Lift", robot=robot)
    else:
        print(f"    Lift: FAILED — {r_lift['status']}")
    _record_flags('lift')

    # ── 4. Move to place pose (above destination) ──
    place_p, place_q = build_place_pose(dest_pos, arm_base)
    place_pose = MPPose(p=place_p, q=place_q)
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_place = planner.plan_pose(place_pose, cq, mask=motion_mask,
                                     planning_time=5.0)
    except TimeoutError:
        r_place = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)
    if r_place['status'] == 'Success':
        print(f"    Place: OK ({r_place['position'].shape[0]} wp)")
        execute_trajectory(r_place['position'], step_fn, GRIPPER_CLOSED,
                           lock_base=used_arm_only, env=env,
                           label="Place", robot=robot)
    else:
        print(f"    Place: FAILED — {r_place['status']}")

    # ── 5. Open gripper to drop ──
    actuate_gripper(step_fn, env, robot, GRIPPER_OPEN, f"{tag} - Drop")
    for _ in range(30):
        step_fn(hold_open())
    _record_flags('release', check_fixture=dest_fixture)

    # ── 6. Return home ──
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    home_qpos = cq.copy()
    home_qpos[3:10] = ARM_HOME
    home_qpos[10:] = 0.0
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_home = planner.plan_qpos([home_qpos], cq, planning_time=5.0)
    except TimeoutError:
        r_home = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)
    if r_home['status'] == 'Success':
        execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                           env=env, label=f"{tag} - Return", robot=robot)
    else:
        print(f"    Return: FAILED, teleporting home")
        robot.set_qpos(torch.tensor(
            home_qpos, dtype=torch.float32).unsqueeze(0))

    wait_until_stable(step_fn,
                      make_action(ARM_HOME, GRIPPER_OPEN,
                                  get_robot_qpos(robot)[:3]),
                      robot, max_steps=100)
    _record_flags('done', check_fixture=dest_fixture)

    # Determine actual outcome from flags
    done_flags = phase_flags[-1]  # 'done' phase
    grasp_flags = next((f for f in phase_flags if f['phase'] == 'grasp'), None)
    grasped = grasp_flags['is_grasped'] if grasp_flags else False
    at_dest = done_flags['obj_at_source']  # checked against dest_fixture
    gripper_clear = done_flags['gripper_far']

    if at_dest and gripper_clear:
        outcome = 'success'
    elif grasped and not at_dest:
        outcome = 'partial'  # picked up but didn't land at dest
    elif not grasped:
        outcome = 'grasp_fail'
    else:
        outcome = 'partial'

    return {'outcome': outcome, 'phases': phase_flags, 'lang': lang}


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoboCasa kitchen grasp test with success flags")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human',
                        choices=['human', 'rgb_array'])
    parser.add_argument('--max-cubes', type=int, default=None,
                        help='Limit to N nearest cubes (default: all)')
    parser.add_argument('--acm', default='strict',
                        choices=['relaxed', 'strict'],
                        help='ACM mode: strict=only relax distant fixtures (default), '
                             'relaxed=ignore all fixtures')
    args = parser.parse_args()

    # --- Environment ---
    print("Creating RoboCasa env...")
    env = gym.make('RoboCasaKitchen-v1', num_envs=1,
                   robot_uids='tidyverse', control_mode='whole_body',
                   render_mode=args.render)
    env.reset(seed=args.seed)

    robot = env.unwrapped.agent.robot
    agent = env.unwrapped.agent
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    if args.render == 'rgb_array':
        base = f'robocasa_success_seed{args.seed}_acm{args.acm}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)

    collision_dir = os.path.join(os.path.dirname(__file__), 'collision_images')
    collision_logger = CollisionLogger(
        robot, scene, env, collision_dir, render_mode=args.render)

    step_label = ["idle"]

    def step_fn(action):
        env.step(action)
        if is_human:
            env.render()
        elif video_writer is not None:
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            video_writer.add_frame(frame.astype(np.uint8))
        collision_logger.check(step_label[0])

    robot_pos = robot.pose.p[0].cpu().numpy()
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot at {robot_pos}, arm base at {arm_base}")

    # --- Collect placements and build pick-place tasks ---
    all_placements = collect_placements(fixtures)
    print(f"\nFound {len(all_placements)} placement locations")

    # Filter to reachable placements and sort by distance
    reachable = []
    for label, pos, ftype, fix_obj in all_placements:
        cube_pos = pos.copy()
        cube_pos[2] += CUBE_HALF + 0.002
        dist = np.linalg.norm(arm_base - cube_pos)
        reachable.append((label, cube_pos, ftype, fix_obj, dist))
    reachable.sort(key=lambda x: x[4])

    # Build task pairs: each task = (source, destination)
    # Pair each placement with a different random destination
    rng = np.random.RandomState(args.seed)
    tasks = []
    for i, (s_label, s_pos, s_ftype, s_fix, s_dist) in enumerate(reachable):
        # Pick a destination that is a different fixture
        candidates = [(j, r) for j, r in enumerate(reachable)
                      if j != i and r[0] != s_label]
        if not candidates:
            continue
        j, (d_label, d_pos, d_ftype, d_fix, d_dist) = candidates[
            rng.randint(len(candidates))]
        tasks.append({
            'src_label': s_label, 'src_pos': s_pos, 'src_ftype': s_ftype,
            'src_fix': s_fix, 'src_dist': s_dist,
            'dst_label': d_label, 'dst_pos': d_pos, 'dst_ftype': d_ftype,
            'dst_fix': d_fix,
        })

    # Sort by source distance, limit
    tasks.sort(key=lambda t: t['src_dist'])
    if args.max_cubes is not None:
        tasks = tasks[:args.max_cubes]

    print(f"\nWill attempt {len(tasks)} pick-place tasks:")
    for i, t in enumerate(tasks):
        print(f"  [{i:2d}] {t['src_label'][:30]:30s} ({t['src_ftype']:12s}) "
              f"-> {t['dst_label'][:30]:30s} ({t['dst_ftype']:12s}) "
              f"dist={t['src_dist']:.2f}m")

    # --- Spawn cubes (one per task, at source) ---
    for i, t in enumerate(tasks):
        color = COLORS[i % len(COLORS)]
        cube_name = f"obj_{i}_{t['src_label']}"
        try:
            actor = spawn_cube(scene, cube_name, t['src_pos'], color)
            t['cube_name'] = cube_name
            t['cube_actor'] = actor
        except Exception as e:
            print(f"  Spawn failed for {t['src_label']}: {e}")
            t['cube_actor'] = None

    # Remove tasks with failed spawns
    tasks = [t for t in tasks if t.get('cube_actor') is not None]

    # --- Stabilize ---
    base_cmd = robot_pos[:3].copy()
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    print("\nStabilizing robot...")
    wait_until_stable(step_fn, hold, robot)

    # --- Setup planner ---
    print("Setting up SapienPlanner...")
    signal.alarm(30)
    try:
        pw = SapienPlanningWorld(scene, [robot._objs[0]])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)
    except TimeoutError:
        print("FATAL: planner setup timed out")
        env.close()
        return
    finally:
        signal.alarm(0)

    cube_names = {t['cube_name'] for t in tasks}
    build_kitchen_acm(pw, planner, cube_names, mode=args.acm,
                      robot_pos=arm_base)

    # --- Pick-place loop ---
    results = {'success': 0, 'partial': 0, 'grasp_fail': 0, 'unreachable': 0, 'error': 0}
    all_results = []

    for ci, t in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(tasks)}] {t['src_label']} ({t['src_ftype']}) "
              f"-> {t['dst_label']} ({t['dst_ftype']})  "
              f"dist={t['src_dist']:.2f}m")

        step_label[0] = f"task {ci+1}/{len(tasks)} {t['src_label']}->{t['dst_label']}"
        try:
            result = attempt_pick_place(
                ci, t['src_pos'], t['src_label'], t['src_ftype'],
                t['dst_pos'], t['dst_label'], t['dst_ftype'],
                robot, planner, pw, step_fn, env, len(tasks),
                scene, agent, t['cube_actor'], t['src_fix'], t['dst_fix'])
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            result = {'outcome': 'error', 'phases': [],
                      'lang': get_pick_place_instruction(
                          t['src_ftype'], t['dst_ftype'], obj_name="cube")}
            try:
                cq = get_robot_qpos(robot)
                cq[3:10] = ARM_HOME
                cq[10:] = 0.0
                robot.set_qpos(torch.tensor(cq, dtype=torch.float32).unsqueeze(0))
            except Exception:
                pass

        outcome = result['outcome']
        results[outcome] = results.get(outcome, 0) + 1
        print(f"  => {outcome.upper()}")

        all_results.append({
            'index': ci,
            'src_label': t['src_label'], 'src_ftype': t['src_ftype'],
            'dst_label': t['dst_label'], 'dst_ftype': t['dst_ftype'],
            'distance': float(t['src_dist']),
            'src_pos': t['src_pos'].tolist(),
            'dst_pos': t['dst_pos'].tolist(),
            'outcome': outcome,
            'lang': result.get('lang', ''),
            'phases': result.get('phases', []),
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}/{len(tasks)}")

    print(f"\nPer-task flag summary:")
    for r in all_results:
        phases = r['phases']
        print(f"[{r['index']+1}/{len(tasks)}] "
              f"{r['src_label']} ({r['src_ftype']}) -> "
              f"{r['dst_label']} ({r['dst_ftype']}) "
              f"=> {r['outcome'].upper()}")
        if r.get('lang'):
            print(f"  lang: {r['lang']}")
        for pf in phases:
            print(f"  {pf['phase']:12s}: {format_flags(pf)}")

    collision_logger.summary()

    # --- Write JSON ---
    json_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(json_dir, exist_ok=True)
    json_path = os.path.join(json_dir, f'success_flags_seed{args.seed}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'seed': args.seed,
            'acm_mode': args.acm,
            'totals': results,
            'tasks': all_results,
        }, f, indent=2)
    print(f"\nJSON results written to: {json_path}")

    if is_human:
        print("\nDone! Close the window to exit.")
        while True:
            try:
                q = get_robot_qpos(robot)
                env.step(make_action(q[3:10], GRIPPER_OPEN, q[:3]))
                env.render()
            except Exception:
                break
    else:
        if video_writer:
            video_writer.close()
        env.close()


if __name__ == '__main__':
    main()
