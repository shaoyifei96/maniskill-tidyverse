#!/usr/bin/env python3
"""RoboCasa kitchen grasp test: place cubes on all surfaces, then pick them up.

Combines fixture enumeration (placing cubes on every available surface) with
the IK-seeded grasp pipeline (pre-grasp → approach → close → lift → drop).

Usage:
    # GUI — watch the robot attempt grasps
    python test_robocasa_grasp.py --render human --seed 0

    # Headless — save video
    python test_robocasa_grasp.py --render rgb_array --seed 0

    # Limit to N nearest cubes
    python test_robocasa_grasp.py --render human --max-cubes 5
"""
import sys, os, signal, argparse, time
import numpy as np
import torch, sapien, cv2
import gymnasium as gym

import maniskill_tidyverse.tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs

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
from maniskill_tidyverse.viz_planning_world import save_planning_world
# Video writer helper
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

# ─── Collision logger ──────────────────────────────────────────────────────────

class CollisionLogger:
    """Detect and log physics collisions (from SAPIEN PhysX contacts).

    Logs each unique collision pair once to terminal, and saves a snapshot image.
    Clears image directory on each run. Filenames include step number for
    cross-referencing with terminal logs.
    """
    def __init__(self, robot, scene, env, img_dir, render_mode='human'):
        self.robot = robot
        self.scene = scene
        self.env = env
        self.img_dir = img_dir
        self.render_mode = render_mode
        # Clear previous collision images
        import shutil
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        # Collect robot link entity names for filtering
        self.robot_entity_names = set()
        for link in robot.get_links():
            self.robot_entity_names.add(link.get_name())
        # Track seen collision pairs (frozenset of (name0, name1))
        self.seen_pairs = set()
        self.collision_count = 0
        self.step_count = 0

    def check(self, step_label=""):
        """Call after each env.step() to check for new collisions."""
        self.step_count += 1
        try:
            contacts = self.scene.get_contacts()
        except Exception:
            return
        for contact in contacts:
            if not contact.points:
                continue
            # Check if any impulse is significant
            impulse = np.sum([pt.impulse for pt in contact.points], axis=0)
            if np.linalg.norm(impulse) < 1e-4:
                continue
            b0 = contact.bodies[0]
            b1 = contact.bodies[1]
            name0 = b0.entity.name if b0.entity else str(b0)
            name1 = b1.entity.name if b1.entity else str(b1)
            # Only care about collisions involving a robot link
            is_robot0 = name0 in self.robot_entity_names
            is_robot1 = name1 in self.robot_entity_names
            if not (is_robot0 or is_robot1):
                continue
            # Skip robot self-collisions
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
                print(f"  ⚠ COLLISION #{self.collision_count} step={self.step_count}: "
                      f"{robot_part} <-> {other_part}  "
                      f"impulse={imp_mag:.4f}  sep={sep:.4f}  "
                      f"[{step_label}]")
                self._save_image(robot_part, other_part)

    def _save_image(self, robot_part, other_part):
        """Save a snapshot of the scene when collision is detected."""
        try:
            frame = self.env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            img = frame.astype(np.uint8)
            # Add collision label
            text = f"COLLISION #{self.collision_count} step={self.step_count}: {robot_part} <-> {other_part}"
            h = img.shape[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.4, h / 800)
            thick = max(1, int(h / 400))
            (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(img, (5, 5), (tw + 15, th_ + 15), (0, 0, 200), -1)
            cv2.putText(img, text, (10, th_ + 10), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            safe_name = (f"collision_{self.collision_count:03d}"
                         f"_step{self.step_count:05d}"
                         f"_{robot_part}_vs_{other_part}.png")
            safe_name = safe_name.replace('/', '_')
            path = os.path.join(self.img_dir, safe_name)
            cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            print(f"    Saved: {path}")
        except Exception as e:
            print(f"    (image save failed: {e})")

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


# ─── Constants ────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.15
CUBE_HALF = 0.02

MASK_ARM_ONLY  = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)

PLANNING_TIMEOUT = 15   # seconds per planning call
IK_TIMEOUT = 8          # seconds for IK pre-check

COLORS = [
    [1.0, 0.0, 0.0, 1], [0.0, 0.8, 0.0, 1], [0.0, 0.3, 1.0, 1],
    [1.0, 0.7, 0.0, 1], [0.8, 0.0, 0.8, 1], [0.0, 0.8, 0.8, 1],
    [1.0, 1.0, 0.0, 1], [1.0, 0.4, 0.7, 1], [0.6, 0.3, 0.0, 1],
    [0.5, 0.5, 0.5, 1],
]


# ─── Timeout handler ─────────────────────────────────────────────────────────

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


# ─── Placement helpers (from fixture enumeration) ────────────────────────────

def local_to_world(fixture, offset):
    """Convert a fixture-local offset to world position."""
    rot_mat = euler2mat(0, 0, fixture.rot)
    return np.array(fixture.pos) + rot_mat @ np.array(offset)


def spawn_cube(scene, name, pos, color):
    """Create a small colored cube at `pos`."""
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
    """Convert reset-region dict to list of (local_offset → world_pos)."""
    results = []
    for rname, region in regions.items():
        offset = np.array(region["offset"], dtype=float)
        wp = local_to_world(fix, offset)
        wp[2] = fix.pos[2] + offset[2]
        results.append((rname, wp))
    return results


def _int_sites_placement(fix, suffix="interior"):
    """Place at the interior center-bottom of a fixture with int_sites."""
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
    """Enumerate all placement surfaces and return [(label, world_pos, fixture_type_str)]."""
    all_placements = []

    for fname, fix in fixtures.items():
        if isinstance(fix, (Floor, Wall)):
            continue

        positions = []  # (region_name, world_pos)
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
            all_placements.append((label, pos, ftype))

    return all_placements


# ─── Motion planning helpers (from test_table_grasp.py) ───────────────────────

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


def check_joint_limits(qpos, joint_limits, joint_names, label=""):
    qi = 0
    for limits, name in zip(joint_limits, joint_names):
        if limits.ndim == 2:
            for d in range(limits.shape[0]):
                if qi >= len(qpos):
                    return
                lo, hi = limits[d, 0], limits[d, 1]
                margin = (hi - lo) * 0.02
                if qpos[qi] <= lo + margin or qpos[qi] >= hi - margin:
                    print(f"    JOINT LIMIT {label}: {name}[{d}] = {qpos[qi]:.4f}")
                qi += 1
        elif limits.ndim == 1 and limits.shape[0] >= 2:
            if qi >= len(qpos):
                return
            qi += 1
        else:
            qi += 1


# ─── ACM builder ─────────────────────────────────────────────────────────────

def _get_object_position(pw, name):
    """Get the world position of a planning-world object."""
    try:
        return np.array(pw.get_object(name).pose.p)
    except Exception:
        return None


def build_kitchen_acm(pw, planner, cube_names, mode='relaxed',
                      robot_pos=None, near_radius=1.5):
    """Configure ACM for collision checking.

    mode='relaxed': relax ALL fixture collisions (planner ignores furniture).
    mode='strict':  only relax fixtures far from robot (>near_radius);
                    nearby fixtures are collision-checked by the planner.
    """
    acm = pw.get_allowed_collision_matrix()
    art_names = pw.get_articulation_names()
    robot_link_names = planner.pinocchio_model.get_link_names()
    robot_art = next(n for n in art_names if 'tidyverse' in n.lower())

    # --- Log what the planner sees ---
    print(f"\n  Planning world contents (ACM mode={mode}):")
    print(f"    Robot links ({len(robot_link_names)}): {robot_link_names[:5]}...")
    print(f"    Articulations ({len(art_names)}):")

    relaxed_arts, checked_arts = [], []
    for an in art_names:
        if an == robot_art:
            print(f"      [ROBOT] {an}")
            continue
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        # Estimate fixture position from its first link
        art_obj = pw.get_articulation(an)
        link_names = art_obj.get_pinocchio_model().get_link_names()
        art_pos = None
        if link_names:
            try:
                art_pos = np.array(art_obj.get_pinocchio_model()
                                   .get_random_configuration()[:3])
            except Exception:
                pass
        # In strict mode, check if fixture is near the robot
        should_relax = True
        if mode == 'strict' and robot_pos is not None and art_pos is not None:
            # Use a rough heuristic: get pose from the FCL objects
            pass  # We'll use object positions below instead

        if mode == 'relaxed':
            relaxed_arts.append(an)
            print(f"      [RELAXED] {an} ({len(fl)} links)")
        else:
            # For strict: we'll decide per-fixture below after checking positions
            checked_arts.append((an, fl))
            print(f"      [PENDING] {an} ({len(fl)} links)")

    obj_names = pw.get_object_names()
    checked_objs = [n for n in obj_names if n in cube_names]

    # Classify static objects
    relaxed_static, checked_static = [], []
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
                # Can't determine distance — keep it checked for safety
                checked_static.append((on, -1))

    print(f"    Static objects ({len(obj_names)} total):")
    print(f"      Cubes (always checked): {len(checked_objs)}")
    if mode == 'relaxed':
        print(f"      ACM-relaxed: {len(relaxed_static)}")
    else:
        print(f"      Collision-checked (near, <{near_radius}m): {len(checked_static)}")
        for on, d in sorted(checked_static, key=lambda x: x[1]):
            print(f"        - {on}  dist={d:.2f}m")
        print(f"      ACM-relaxed (far): {len(relaxed_static)}")

    # --- Apply ACM ---
    if mode == 'relaxed':
        # Relax ALL fixture articulations
        for an in art_names:
            if an == robot_art:
                continue
            fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
            for rl in robot_link_names:
                for f in fl:
                    acm.set_entry(rl, f, True)
        # Relax ALL non-cube static objects
        for on in relaxed_static:
            for rl in robot_link_names:
                acm.set_entry(rl, on, True)
    else:
        # Strict: relax fixture articulations (too many false collisions
        # from articulated fixture meshes), but keep nearby static objects
        # as collision obstacles for the planner.
        for an in art_names:
            if an == robot_art:
                continue
            fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
            for rl in robot_link_names:
                for f in fl:
                    acm.set_entry(rl, f, True)
        # Only relax FAR static objects
        for on in relaxed_static:
            for rl in robot_link_names:
                acm.set_entry(rl, on, True)
        # Nearby static objects are NOT relaxed — planner will avoid them

    # Check initial collisions
    sync_planner(planner)
    collisions = pw.check_collision()
    if collisions:
        print(f"\n  Initial planner collisions ({len(collisions)}):")
        for c in collisions:
            print(f"    {c.link_name1}({c.object_name1}) <-> "
                  f"{c.link_name2}({c.object_name2})")
    else:
        print(f"\n  No initial planner collisions")

    n_relaxed_art = len(art_names) - 1 if mode == 'relaxed' else len(art_names) - 1
    print(f"\n  ACM summary: {n_relaxed_art} fixture articulations relaxed, "
          f"{len(relaxed_static)} static objects relaxed, "
          f"{len(checked_objs) + len(checked_static)} objects collision-checked")


# ─── Grasp strategy ──────────────────────────────────────────────────────────

def build_grasp_poses(cube_pos, arm_base):
    """Compute grasp poses for a cube. Returns [(name, pos, quat_wxyz)]."""
    yaw = np.arctan2(cube_pos[1] - arm_base[1], cube_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    front_rot = R.from_euler('yz', [np.pi / 2, yaw])
    return [
        ('Top-Down',
         cube_pos + [0, 0, 0],
         [0, 1, 0, 0]),
        ('Front',
         cube_pos + [-0.06 * cos_y, -0.06 * sin_y, 0.08],
         list(front_rot.as_quat()[[3, 0, 1, 2]])),
        ('Angled45',
         cube_pos + [-0.02 * cos_y, -0.02 * sin_y, 0.02],
         list(euler2quat(0, 3 * np.pi / 4, yaw))),
    ]


def select_strategies(ftype, label):
    """Return ordered list of grasp strategy names to try, based on fixture type."""
    is_enclosed = 'interior' in label
    if is_enclosed:
        # Cabinet/microwave interior — top-down is blocked by ceiling
        return ['Front', 'Angled45']
    elif ftype in ('Stove', 'Stovetop'):
        return ['Top-Down']
    else:
        # Open surfaces: counters, cabinet tops, sink basin
        return ['Top-Down', 'Angled45', 'Front']


# ─── Single grasp attempt ────────────────────────────────────────────────────

def attempt_grasp(cube_idx, cube_name, cube_pos, label, ftype,
                  robot, planner, pw, step_fn, env, total, timings,
                  viz_dir=None, step_label=None):
    """Full pick cycle for one cube. Returns outcome string."""
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    strategies = select_strategies(ftype, label)
    all_grasps = build_grasp_poses(cube_pos, arm_base)
    ordered = [g for s in strategies for g in all_grasps if g[0] == s]
    if not ordered:
        ordered = all_grasps

    def _snap(stage_name, gname_s=""):
        if viz_dir:
            sync_planner(planner)
            slug = label.lower().replace(' ', '_').replace('/', '_')[:40]
            g_slug = gname_s.lower().replace('-', '_').replace(' ', '_')
            save_planning_world(
                pw, os.path.join(viz_dir, f"{cube_idx}_{slug}_{g_slug}_{stage_name}"))

    def _set_label(phase):
        if step_label is not None:
            step_label[0] = f"[{cube_idx+1}/{total}] {label} | {phase}"

    for gname, target_p, target_q in ordered:
        tag = f"[{cube_idx+1}/{total}] {label} ({gname})"
        _set_label(f"IK ({gname})")
        print(f"\n  --- {tag} ---")
        target_q_arr = np.array(target_q)

        approach_pose = MPPose(p=np.array(target_p), q=target_q_arr)
        pre_pose = MPPose(p=np.array(target_p) + [0, 0, PRE_GRASP_HEIGHT],
                          q=target_q_arr)

        sync_planner(planner)
        cq = get_robot_qpos(robot)

        # Transform to base-relative for IK
        approach_base = planner._transform_goal_to_wrt_base(approach_pose)
        pre_base = planner._transform_goal_to_wrt_base(pre_pose)

        # 1. Solve grasp IK (arm-only, then whole-body fallback)
        q_grasp = None
        grasp_mask = None
        for mask_name, mask in [("arm-only", MASK_ARM_ONLY),
                                ("whole-body", MASK_WHOLE_BODY)]:
            t0 = time.time()
            signal.alarm(IK_TIMEOUT)
            try:
                status, solutions = planner.IK(
                    approach_base, cq, mask=mask, n_init_qpos=40,
                    return_closest=True)
            except TimeoutError:
                dt = time.time() - t0
                print(f"    IK ({mask_name}): TIMEOUT  [{dt:.2f}s]")
                timings['ik'] += dt
                continue
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            timings['ik'] += dt
            if solutions is not None:
                q_grasp = solutions
                grasp_mask = mask
                print(f"    Grasp IK ({mask_name}): OK  [{dt:.2f}s]")
                break
            else:
                print(f"    Grasp IK ({mask_name}): no solution  [{dt:.2f}s]")

        if q_grasp is None:
            print(f"    Grasp IK: FAILED for {gname} — trying next strategy")
            continue

        # 2. Solve pre-grasp IK seeded from grasp solution
        t0 = time.time()
        signal.alarm(IK_TIMEOUT)
        try:
            _, pregrasp_sols = planner.IK(
                pre_base, q_grasp, mask=grasp_mask, n_init_qpos=40,
                return_closest=True)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Pre-grasp IK: TIMEOUT  [{dt:.2f}s]")
            timings['ik'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['ik'] += dt
        if pregrasp_sols is None:
            print(f"    Pre-grasp IK: FAILED  [{dt:.2f}s]")
            continue
        print(f"    Pre-grasp IK: OK  [{dt:.2f}s]")

        # 3. Plan path: current → pre-grasp
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            result = planner.plan_qpos([pregrasp_sols], cq, planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Pre-grasp path: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if result['status'] != 'Success':
            print(f"    Pre-grasp path: FAILED — {result['status']}  [{dt:.2f}s]")
            continue
        print(f"    Pre-grasp path: OK ({result['position'].shape[0]} wp)  [{dt:.2f}s]")

        used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
        motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

        # Execute pre-grasp
        _set_label(f"Pre-grasp ({gname})")
        t0 = time.time()
        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, env=env,
                           label="Pre-grasp", robot=robot)
        timings['exec'] += time.time() - t0

        def hold_open():
            q = get_robot_qpos(robot)
            return make_action(q[3:10], GRIPPER_OPEN, q[:3])

        def hold_closed():
            q = get_robot_qpos(robot)
            return make_action(q[3:10], GRIPPER_CLOSED, q[:3])

        t0 = time.time()
        pause_with_label(env, step_fn, hold_open(), f"{tag} - Pre-grasp")
        timings['pause'] += time.time() - t0
        _snap("1_pregrasp", gname)

        # 4. Approach
        _set_label(f"Approach ({gname})")
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_app = planner.plan_pose(approach_pose, cq, mask=motion_mask,
                                       planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Approach: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_app['status'] != 'Success':
            print(f"    Approach: FAILED — {r_app['status']}  [{dt:.2f}s]")
            continue
        print(f"    Approach: OK ({r_app['position'].shape[0]} wp)  [{dt:.2f}s]")

        t0 = time.time()
        execute_trajectory(r_app['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, env=env,
                           label="Approach", robot=robot)
        timings['exec'] += time.time() - t0

        t0 = time.time()
        pause_with_label(env, step_fn, hold_open(), f"{tag} - Approach")
        timings['pause'] += time.time() - t0
        _snap("2_approach", gname)

        # 5. Close gripper
        _set_label(f"Grasping ({gname})")
        t0 = time.time()
        actuate_gripper(step_fn, env, robot, GRIPPER_CLOSED,
                        f"{tag} - Closing")
        pause_with_label(env, step_fn, hold_closed(), f"{tag} - Grasped")
        timings['gripper'] += time.time() - t0
        _snap("3_grasped", gname)

        # 6. Lift
        _set_label(f"Lifting ({gname})")
        lift_pose = MPPose(p=np.array(target_p) + [0, 0, LIFT_HEIGHT],
                           q=target_q_arr)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_lift = planner.plan_pose(lift_pose, cq, mask=motion_mask,
                                        planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Lift: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            # still got grasp — partial success
            actuate_gripper(step_fn, env, robot, GRIPPER_OPEN,
                            f"{tag} - Drop (lift failed)")
            return 'partial'
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_lift['status'] == 'Success':
            print(f"    Lift: OK ({r_lift['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_lift['position'], step_fn, GRIPPER_CLOSED,
                               lock_base=used_arm_only, env=env,
                               label="Lift", robot=robot)
            timings['exec'] += time.time() - t0
            t0 = time.time()
            pause_with_label(env, step_fn, hold_closed(), f"{tag} - Lifted")
            timings['pause'] += time.time() - t0
            _snap("4_lifted", gname)
        else:
            print(f"    Lift: FAILED — {r_lift['status']}  [{dt:.2f}s]")

        # 7. Drop
        _set_label(f"Dropping ({gname})")
        t0 = time.time()
        actuate_gripper(step_fn, env, robot, GRIPPER_OPEN,
                        f"{tag} - Dropping")
        for _ in range(30):
            step_fn(hold_open())
        timings['gripper'] += time.time() - t0

        # 8. Return to home
        _set_label(f"Return home ({gname})")
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        home_qpos = cq.copy()
        home_qpos[3:10] = ARM_HOME
        home_qpos[10:] = 0.0
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_home = planner.plan_qpos([home_qpos], cq, planning_time=5.0)
        except TimeoutError:
            r_home = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_home['status'] == 'Success':
            print(f"    Return: OK ({r_home['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                               env=env, label=f"{tag} - Return", robot=robot)
            timings['exec'] += time.time() - t0
        else:
            print(f"    Return: FAILED, teleporting to home  [{dt:.2f}s]")
            robot.set_qpos(torch.tensor(
                home_qpos, dtype=torch.float32).unsqueeze(0))

        t0 = time.time()
        wait_until_stable(step_fn,
                          make_action(ARM_HOME, GRIPPER_OPEN,
                                      get_robot_qpos(robot)[:3]),
                          robot, max_steps=100)
        timings['settle'] += time.time() - t0
        return 'success'

    return 'unreachable'


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoboCasa kitchen grasp test")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human',
                        choices=['human', 'rgb_array'])
    parser.add_argument('--max-cubes', type=int, default=None,
                        help='Limit to N nearest cubes (default: all)')
    parser.add_argument('--acm', default='relaxed',
                        choices=['relaxed', 'strict'],
                        help='ACM mode: relaxed=ignore all fixtures, '
                             'strict=only relax distant fixtures')
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Save planning-world collision meshes (glb) '
                             'at each grasp stage to this directory')
    args = parser.parse_args()

    # --- Environment ---
    t_total = time.time()
    t0 = time.time()
    print("Creating RoboCasa env...")
    env = gym.make('RoboCasaKitchen-v1', num_envs=1,
                   robot_uids='tidyverse', control_mode='whole_body',
                   render_mode=args.render)
    env.reset(seed=args.seed)
    t_env = time.time() - t0
    print(f"  env setup: {t_env:.2f}s")

    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    # Reposition render camera: higher up, tilted down toward robot
    from mani_skill.utils import sapien_utils as _su
    _rpos = robot.pose.p[0].cpu().numpy()
    _cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]  # behind + high
    _cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]  # look ahead and down
    _cam_pose = _su.look_at(_cam_eye, _cam_target)
    # Convert ManiSkill batched Pose to plain sapien.Pose
    _p = _cam_pose.raw_pose[0].cpu().numpy()
    _sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
    for cam in env.unwrapped._human_render_cameras.values():
        cam.camera.set_local_pose(_sapien_pose)

    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    if args.render == 'rgb_array':
        # Find next available run number to avoid overwriting
        base = f'robocasa_grasp_seed{args.seed}_acm{args.acm}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)

    collision_dir = os.path.join(os.path.dirname(__file__), 'collision_images')
    collision_logger = CollisionLogger(
        robot, scene, env, collision_dir, render_mode=args.render)

    step_label = ["idle"]  # mutable for closure

    def _burn_label(frame, text):
        """Burn a text banner onto the top of a frame."""
        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600)
        thick = max(1, int(h / 300))
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 20, th_ + 16), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, th_ + 8), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def step_fn(action):
        env.step(action)
        if is_human:
            env.render()
        elif video_writer is not None:
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frame = frame.astype(np.uint8).copy()
            _burn_label(frame, step_label[0])
            video_writer.add_frame(frame)
        collision_logger.check(step_label[0])

    robot_pos = robot.pose.p[0].cpu().numpy()
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot at {robot_pos}, arm base at {arm_base}")

    # --- Collect and spawn cubes on all surfaces ---
    all_placements = collect_placements(fixtures)
    print(f"\nFound {len(all_placements)} placement locations")

    spawned = []  # (cube_name, cube_pos, label, ftype, dist)
    for i, (label, pos, ftype) in enumerate(all_placements):
        color = COLORS[i % len(COLORS)]
        cube_pos = pos.copy()
        cube_pos[2] += CUBE_HALF + 0.002
        cube_name = f"obj_{i}_{label}"
        try:
            spawn_cube(scene, cube_name, cube_pos, color)
            dist = np.linalg.norm(arm_base - cube_pos)
            spawned.append((cube_name, cube_pos, label, ftype, dist))
        except Exception as e:
            print(f"  Spawn failed for {label}: {e}")

    # Sort by distance (nearest first)
    spawned.sort(key=lambda x: x[4])
    if args.max_cubes is not None:
        spawned = spawned[:args.max_cubes]

    print(f"\nWill attempt {len(spawned)} cubes (sorted by distance):")
    for i, (name, pos, label, ftype, dist) in enumerate(spawned):
        print(f"  [{i:2d}] {label:45s} dist={dist:.2f}m  ({ftype})")

    # --- Stabilize ---
    # Use base qpos (not robot.pose.p) as hold target — the PD controller
    # drives base joints, which are relative to the root pose.
    base_cmd = get_robot_qpos(robot)[:3].copy()
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    step_label[0] = "Stabilizing"
    print("\nStabilizing robot...")
    t0 = time.time()
    wait_until_stable(step_fn, hold, robot)
    t_stabilize = time.time() - t0
    print(f"  stabilize: {t_stabilize:.2f}s")

    # --- Setup planner ---
    print("Setting up SapienPlanner...")
    t0 = time.time()
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
    t_planner = time.time() - t0
    print(f"  planner setup: {t_planner:.2f}s")

    cube_names = {s[0] for s in spawned}
    t0 = time.time()
    build_kitchen_acm(pw, planner, cube_names, mode=args.acm,
                      robot_pos=arm_base)
    t_acm = time.time() - t0
    print(f"  ACM build: {t_acm:.2f}s")

    # --- Snapshot initial planning world ---
    if args.viz_dir:
        sync_planner(planner)
        save_planning_world(pw, os.path.join(args.viz_dir, "initial_home"))

    # --- Grasp loop ---
    timings = {'ik': 0.0, 'planning': 0.0, 'exec': 0.0,
               'pause': 0.0, 'gripper': 0.0, 'settle': 0.0}
    results = {'success': 0, 'partial': 0, 'unreachable': 0, 'error': 0}
    for ci, (cube_name, cube_pos, label, ftype, dist) in enumerate(spawned):
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(spawned)}] {label} ({ftype})  "
              f"dist={dist:.2f}m  pos={cube_pos}")

        step_label[0] = f"grasp {ci+1}/{len(spawned)} {label}"
        try:
            outcome = attempt_grasp(ci, cube_name, cube_pos, label, ftype,
                                    robot, planner, pw, step_fn, env,
                                    len(spawned), timings,
                                    viz_dir=args.viz_dir,
                                    step_label=step_label)
        except Exception as e:
            print(f"  ERROR: {e}")
            outcome = 'error'
            # Try to recover: go home
            try:
                cq = get_robot_qpos(robot)
                cq[3:10] = ARM_HOME
                cq[10:] = 0.0
                robot.set_qpos(torch.tensor(cq, dtype=torch.float32).unsqueeze(0))
            except Exception:
                pass

        results[outcome] = results.get(outcome, 0) + 1
        print(f"  => {outcome.upper()}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}/{len(spawned)}")

    collision_logger.summary()

    t_total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print("TIMING BREAKDOWN:")
    print(f"  env setup:       {t_env:7.2f}s")
    print(f"  stabilize:       {t_stabilize:7.2f}s")
    print(f"  planner setup:   {t_planner:7.2f}s")
    print(f"  ACM build:       {t_acm:7.2f}s")
    print(f"  --- per-grasp ---")
    print(f"  IK solving:      {timings['ik']:7.2f}s")
    print(f"  path planning:   {timings['planning']:7.2f}s")
    print(f"  traj execution:  {timings['exec']:7.2f}s")
    print(f"  pauses:          {timings['pause']:7.2f}s")
    print(f"  gripper:         {timings['gripper']:7.2f}s")
    print(f"  settle:          {timings['settle']:7.2f}s")
    grasp_total = sum(timings.values())
    accounted = t_env + t_stabilize + t_planner + t_acm + grasp_total
    print(f"  --- totals ---")
    print(f"  grasp phases:    {grasp_total:7.2f}s")
    print(f"  accounted:       {accounted:7.2f}s")
    print(f"  wall clock:      {t_total_elapsed:7.2f}s")
    print(f"  unaccounted:     {t_total_elapsed - accounted:7.2f}s")

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
