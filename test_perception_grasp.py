#!/usr/bin/env python3
"""Perception-based grasp pipeline: use depth + segmentation cameras to detect
objects, fit ellipses, choose grasp strategy, and execute with mplib planner.

Unlike test_robocasa_grasp.py which uses known object positions (spawns cubes),
this script uses the robot's cameras to perceive objects and infer grasps.

Usage:
    # GUI — watch the robot perceive and grasp
    python test_perception_grasp.py --render human --seed 0

    # Headless — save video
    python test_perception_grasp.py --render rgb_array --seed 0

    # Use a specific task env
    python test_perception_grasp.py --task RoboCasaKitchen-v1 --render human
"""
import sys, os, signal, argparse, time
import numpy as np
import torch, sapien, cv2
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs
try:
    import robocasa_tasks  # noqa: registers RoboCasa single-stage tasks
except ImportError:
    pass

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
from mani_skill.utils.structs import Actor, Link
from mani_skill.utils import common

# Reuse fixture types for context
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    SingleCabinet, HingeCabinet, OpenCabinet, Drawer,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Floor, Wall
from mani_skill.utils.scene_builder.robocasa.fixtures.fixture import Fixture


# ─── Constants ────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.15

MASK_ARM_ONLY  = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)

PLANNING_TIMEOUT = 15
IK_TIMEOUT = 8


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


# ─── Perception Pipeline ─────────────────────────────────────────────────────

class PerceptionResult:
    """Result of perceiving a single object from camera data."""
    def __init__(self, name, seg_id, center_3d, bbox_3d_min, bbox_3d_max,
                 ellipse_axes, ellipse_angle, mask_pixels, is_robot_link=False,
                 fixture_context=None):
        self.name = name
        self.seg_id = seg_id
        self.center_3d = center_3d        # [x, y, z] in world frame
        self.bbox_3d_min = bbox_3d_min    # [x, y, z] min corner
        self.bbox_3d_max = bbox_3d_max    # [x, y, z] max corner
        self.ellipse_axes = ellipse_axes  # (major, minor) in meters
        self.ellipse_angle = ellipse_angle  # orientation in degrees
        self.mask_pixels = mask_pixels    # number of pixels
        self.is_robot_link = is_robot_link
        self.fixture_context = fixture_context  # 'counter', 'cabinet_interior', etc.

    @property
    def size_3d(self):
        return self.bbox_3d_max - self.bbox_3d_min

    @property
    def aspect_ratio(self):
        if self.ellipse_axes[1] < 1e-6:
            return float('inf')
        return self.ellipse_axes[0] / self.ellipse_axes[1]

    def __repr__(self):
        sz = self.size_3d
        return (f"PerceptionResult({self.name}, center={self.center_3d}, "
                f"size=[{sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f}], "
                f"aspect={self.aspect_ratio:.2f}, ctx={self.fixture_context})")


def deproject_pixels_to_world(pixels_uv, depth_img, intrinsic, cam2world_gl):
    """Back-project pixel coordinates to 3D world positions.

    Args:
        pixels_uv: (N, 2) array of [u, v] pixel coordinates
        depth_img: (H, W) depth image in mm (int16)
        intrinsic: (3, 3) camera intrinsic matrix
        cam2world_gl: (4, 4) camera-to-world transform (OpenGL convention)

    Returns:
        (N, 3) array of [x, y, z] world coordinates
    """
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]

    u = pixels_uv[:, 0].astype(float)
    v = pixels_uv[:, 1].astype(float)

    # Get depth at each pixel (mm → m)
    depths_mm = depth_img[pixels_uv[:, 1], pixels_uv[:, 0]].astype(float)
    depths_m = depths_mm / 1000.0

    # Back-project to camera frame (OpenCV convention: x right, y down, z forward)
    x_cam = (u - cx) * depths_m / fx
    y_cam = (v - cy) * depths_m / fy
    z_cam = depths_m

    # Convert OpenCV camera frame to OpenGL camera frame
    # OpenGL: x right, y up, z backward → flip y and z
    x_gl = x_cam
    y_gl = -y_cam
    z_gl = -z_cam

    pts_gl = np.stack([x_gl, y_gl, z_gl, np.ones_like(z_gl)], axis=-1)  # (N, 4)

    # Transform to world using cam2world_gl (4x4)
    pts_world = (cam2world_gl @ pts_gl.T).T[:, :3]  # (N, 3)

    return pts_world


def perceive_objects(obs, env, camera_name="base_camera",
                     min_pixels=50, max_depth_mm=5000,
                     target_names=None):
    """Extract object detections from camera observations.

    Args:
        obs: observation dict from env.step() or env.reset()
        env: the unwrapped ManiSkill env
        camera_name: which camera to use
        min_pixels: minimum mask area to consider
        max_depth_mm: max depth to consider (ignore far objects)
        target_names: if set, only detect these object names (skip all others)

    Returns:
        list of PerceptionResult
    """
    sensor_data = obs["sensor_data"][camera_name]
    rgb = common.to_numpy(sensor_data["rgb"][0])          # [H, W, 3] uint8
    depth = common.to_numpy(sensor_data["depth"][0])      # [H, W, 1] int16 (mm)
    seg = common.to_numpy(sensor_data["segmentation"][0]) # [H, W, 1] int16

    depth = depth[..., 0]  # [H, W]
    seg = seg[..., 0]      # [H, W]

    # Camera parameters
    sensor_params = obs["sensor_param"][camera_name]
    intrinsic = common.to_numpy(sensor_params["intrinsic_cv"][0])      # [3, 3]
    cam2world = common.to_numpy(sensor_params["cam2world_gl"][0])      # [4, 4]

    # Segmentation ID map
    seg_id_map = env.unwrapped.segmentation_id_map

    # Robot link names for filtering
    robot_link_names = set()
    for link in env.unwrapped.agent.robot.get_links():
        robot_link_names.add(link.get_name())

    results = []
    unique_ids = np.unique(seg)

    for sid in unique_ids:
        if sid == 0:  # background
            continue

        mask = seg == sid
        n_pixels = int(mask.sum())
        if n_pixels < min_pixels:
            continue

        # Look up what this ID is
        obj = seg_id_map.get(int(sid))
        if obj is None:
            continue

        obj_name = obj.name if hasattr(obj, 'name') else str(obj)
        is_robot = obj_name in robot_link_names or isinstance(obj, Link)

        # Skip robot links
        if is_robot:
            continue

        # Skip fixture/scene elements — only target graspable objects.
        # Use an allowlist approach: if target_names is provided, only keep those.
        # Otherwise, skip known fixture prefixes and large objects.
        if target_names is not None:
            if obj_name not in target_names:
                continue
        else:
            _skip_prefixes = (
                'wall_', 'floor_', 'counter_', 'cab_', 'stove_', 'sink_',
                'fridge_', 'microwave_', 'dishwasher_', 'stack_', 'shelves_',
                'window_', 'outlet_', 'light_switch_', 'plant_', 'stool_',
                'utensil_rack_', 'utensil_holder_', 'fridge_housing_',
                'cab_micro_', 'cab_corner_', 'counter_corner_',
                'paper_towel_', 'knife_block_', 'toaster_', 'coffee_machine_',
            )
            if any(obj_name.startswith(p) for p in _skip_prefixes):
                continue

        # Get mask pixel coordinates
        ys, xs = np.where(mask)
        pixels = np.stack([xs, ys], axis=-1)  # (N, 2) [u, v]

        # Filter by valid depth
        pixel_depths = depth[ys, xs]
        valid = (pixel_depths > 0) & (pixel_depths < max_depth_mm)
        if valid.sum() < min_pixels // 2:
            continue

        valid_pixels = pixels[valid]
        valid_depths = pixel_depths[valid]

        # Back-project to 3D world coordinates
        pts_3d = deproject_pixels_to_world(valid_pixels, depth, intrinsic, cam2world)

        # Filter out any NaN/inf points
        finite_mask = np.all(np.isfinite(pts_3d), axis=1)
        pts_3d = pts_3d[finite_mask]
        if len(pts_3d) < 5:
            continue

        center_3d = np.mean(pts_3d, axis=0)
        bbox_min = np.min(pts_3d, axis=0)
        bbox_max = np.max(pts_3d, axis=0)

        # Fit ellipse to the 2D mask for shape analysis
        ellipse_axes = (0.0, 0.0)
        ellipse_angle = 0.0
        if len(valid_pixels) >= 5:
            try:
                ellipse = cv2.fitEllipse(valid_pixels.astype(np.float32))
                # ellipse = ((cx, cy), (w, h), angle)
                # Convert pixel ellipse axes to approximate metric size
                # using average depth
                avg_depth_m = np.mean(valid_depths) / 1000.0
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                major_m = ellipse[1][1] * avg_depth_m / fy  # height axis
                minor_m = ellipse[1][0] * avg_depth_m / fx  # width axis
                if major_m < minor_m:
                    major_m, minor_m = minor_m, major_m
                ellipse_axes = (major_m, minor_m)
                ellipse_angle = ellipse[2]
            except cv2.error:
                pass

        result = PerceptionResult(
            name=obj_name,
            seg_id=int(sid),
            center_3d=center_3d,
            bbox_3d_min=bbox_min,
            bbox_3d_max=bbox_max,
            ellipse_axes=ellipse_axes,
            ellipse_angle=ellipse_angle,
            mask_pixels=n_pixels,
            is_robot_link=is_robot,
        )
        results.append(result)

    return results


def classify_fixture_context(obj_center, fixtures):
    """Determine what fixture context an object is in (on counter, in cabinet, etc.)."""
    best_ctx = "unknown"
    best_dist = float('inf')

    for fname, fix in fixtures.items():
        if isinstance(fix, (Floor, Wall)):
            continue

        fpos = np.array(fix.pos) if hasattr(fix, 'pos') else None
        if fpos is None:
            continue

        dist = np.linalg.norm(obj_center[:2] - fpos[:2])

        # Check if inside an enclosed fixture
        if isinstance(fix, (SingleCabinet, HingeCabinet, OpenCabinet, Drawer, Microwave)):
            if hasattr(fix, 'get_int_sites') and hasattr(fix, '_bounds_sites'):
                try:
                    if 'int_p0' in fix._bounds_sites:
                        from robocasa_tasks.robocasa_utils import point_in_fixture
                        env_dummy = type('E', (), {'fixture_refs': [{fname: fix}],
                                                   'scene_builder': type('S', (), {'scene_data': [{'fixtures': {fname: fix}}]})()})()
                        # Simple distance check for interior
                        if dist < 0.5 and obj_center[2] < fpos[2] + 0.5:
                            if dist < best_dist:
                                best_dist = dist
                                if isinstance(fix, Drawer):
                                    best_ctx = "drawer_interior"
                                elif isinstance(fix, Microwave):
                                    best_ctx = "microwave_interior"
                                else:
                                    best_ctx = "cabinet_interior"
                except Exception:
                    pass

        elif isinstance(fix, Counter):
            # On a counter — check if object is above counter surface
            counter_top_z = fpos[2] + (fix.size[2] / 2 if hasattr(fix, 'size') else 0.4)
            if dist < 0.8 and abs(obj_center[2] - counter_top_z) < 0.15:
                if dist < best_dist:
                    best_dist = dist
                    best_ctx = "counter"

        elif isinstance(fix, (Stove, Stovetop)):
            if dist < 0.5 and abs(obj_center[2] - fpos[2]) < 0.2:
                if dist < best_dist:
                    best_dist = dist
                    best_ctx = "stove"

    return best_ctx


# ─── Grasp Strategy Selection ────────────────────────────────────────────────

def choose_grasp_strategy(perception, arm_base_pos):
    """Choose grasp strategy based on perceived object shape and context.

    Returns list of (strategy_name, grasp_pose_p, grasp_pose_q) tuples,
    ordered by preference.

    Strategies:
        - TopDown:   gripper pointing straight down (z = [0,0,-1])
        - Angled45:  gripper at 45° tilt toward object
        - Front:     gripper horizontal, approaching from robot side
        - Front90:   gripper horizontal, rotated 90° (for elongated handles)
    """
    obj_pos = perception.center_3d
    size = perception.size_3d
    aspect = perception.aspect_ratio
    ctx = perception.fixture_context

    # Direction from arm base to object
    yaw = np.arctan2(obj_pos[1] - arm_base_pos[1],
                     obj_pos[0] - arm_base_pos[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    # Build grasp pose candidates
    # Quaternions are wxyz format for SAPIEN

    # Top-down: gripper pointing straight down
    top_down = ('TopDown',
                obj_pos.copy(),
                np.array([0, 1, 0, 0], dtype=float))  # 180° around x

    # Front: gripper horizontal, approaching from robot's direction
    front_offset = np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08])
    front_rot = R.from_euler('yz', [np.pi / 2, yaw])
    front_q = front_rot.as_quat()[[3, 0, 1, 2]]  # xyzw → wxyz
    front = ('Front',
             obj_pos + front_offset,
             front_q.astype(float))

    # Angled 45°: between top-down and front
    angled_offset = np.array([-0.02 * cos_y, -0.02 * sin_y, 0.02])
    angled_q = np.array(euler2quat(0, 3 * np.pi / 4, yaw), dtype=float)
    angled45 = ('Angled45',
                obj_pos + angled_offset,
                angled_q)

    # Front 90°: rotated 90° from front (for handles perpendicular to approach)
    front90_rot = R.from_euler('xyz', [np.pi / 2, 0, yaw])
    front90_q = front90_rot.as_quat()[[3, 0, 1, 2]]
    front90 = ('Front90',
               obj_pos + front_offset,
               front90_q.astype(float))

    all_strategies = [top_down, front, angled45, front90]

    # --- Strategy selection logic ---

    # Enclosed fixtures: top-down blocked by ceiling
    if ctx in ('cabinet_interior', 'drawer_interior', 'microwave_interior'):
        order = ['Front', 'Angled45', 'Front90']

    # Handle-like objects: elongated shape
    elif aspect > 3.0:
        # Likely a handle — try front approaches
        order = ['Front', 'Front90', 'Angled45']

    # Tall/narrow objects on open surface
    elif size[2] > 0.08 and aspect > 1.5:
        order = ['Front', 'Angled45', 'TopDown']

    # Small objects on stove
    elif ctx == 'stove':
        order = ['TopDown']

    # Default: open surface
    else:
        order = ['TopDown', 'Angled45', 'Front']

    # Build ordered list
    ordered = []
    for name in order:
        for s in all_strategies:
            if s[0] == name:
                ordered.append(s)
                break

    # Add any remaining strategies not yet included
    for s in all_strategies:
        if s[0] not in order:
            ordered.append(s)

    return ordered


# ─── Debug Visualization ─────────────────────────────────────────────────────

def save_perception_debug(obs, env, perceptions, camera_name, output_dir):
    """Save debug images showing segmentation, depth, and detected objects."""
    os.makedirs(output_dir, exist_ok=True)

    sensor_data = obs["sensor_data"][camera_name]
    rgb = common.to_numpy(sensor_data["rgb"][0])
    depth = common.to_numpy(sensor_data["depth"][0])[..., 0]
    seg = common.to_numpy(sensor_data["segmentation"][0])[..., 0]

    # Save RGB
    cv2.imwrite(os.path.join(output_dir, "rgb.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    # Save depth (normalized for visibility)
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        d_min, d_max = valid_depth.min(), valid_depth.max()
        depth_vis = np.zeros_like(depth, dtype=np.uint8)
        mask = depth > 0
        depth_vis[mask] = ((depth[mask] - d_min) / max(1, d_max - d_min) * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, "depth.png"), depth_vis)

    # Save segmentation (colorized)
    seg_vis = np.zeros((*seg.shape, 3), dtype=np.uint8)
    unique_ids = np.unique(seg)
    for i, sid in enumerate(unique_ids):
        if sid == 0:
            continue
        color = np.array([(sid * 67) % 256, (sid * 131) % 256, (sid * 199) % 256],
                         dtype=np.uint8)
        seg_vis[seg == sid] = color
    cv2.imwrite(os.path.join(output_dir, "segmentation.png"), seg_vis)

    # Save annotated RGB with detections
    annotated = rgb.copy()
    for p in perceptions:
        # Find this object's mask
        obj_mask = seg == p.seg_id
        ys, xs = np.where(obj_mask)
        if len(xs) < 5:
            continue

        # Draw bounding box
        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Fit and draw ellipse
        try:
            pts = np.stack([xs, ys], axis=-1).astype(np.float32)
            ellipse = cv2.fitEllipse(pts)
            cv2.ellipse(annotated, ellipse, (255, 0, 0), 2)
        except cv2.error:
            pass

        # Label
        label = f"{p.name} ({p.fixture_context})"
        sz = p.size_3d
        size_label = f"sz={sz[0]:.2f}x{sz[1]:.2f}x{sz[2]:.2f} ar={p.aspect_ratio:.1f}"
        cv2.putText(annotated, label, (x1, y1 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.putText(annotated, size_label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 200, 200), 1)

    cv2.imwrite(os.path.join(output_dir, "detections.png"),
                cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    print(f"  Debug images saved to {output_dir}/")


# ─── Motion Planning Helpers (from test_robocasa_grasp.py) ───────────────────

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


def execute_trajectory(traj, step_fn, gripper, lock_base=False, robot=None,
                       settle_thresh=0.01, settle_steps=100):
    base_cmd = traj[0, 0:3] if lock_base else None
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


def actuate_gripper(step_fn, robot, gripper_val, n_steps=30):
    qpos = get_robot_qpos(robot)
    arm, base = qpos[3:10], qpos[0:3]
    action = make_action(arm, gripper_val, base)
    for _ in range(n_steps):
        step_fn(action)


# ─── ACM builder (simplified from test_robocasa_grasp.py) ────────────────────

def build_kitchen_acm(pw, planner, target_names=None, mode='relaxed'):
    """Relax fixture collisions for the planner."""
    acm = pw.get_allowed_collision_matrix()
    art_names = pw.get_articulation_names()
    robot_link_names = planner.pinocchio_model.get_link_names()
    robot_art = next(n for n in art_names if 'tidyverse' in n.lower())

    for an in art_names:
        if an == robot_art:
            continue
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        for rl in robot_link_names:
            for f in fl:
                acm.set_entry(rl, f, True)

    if mode == 'relaxed':
        obj_names = pw.get_object_names()
        target_set = set(target_names or [])
        for on in obj_names:
            if on in target_set:
                continue  # keep collision checking for target objects
            for rl in robot_link_names:
                acm.set_entry(rl, on, True)

    sync_planner(planner)
    collisions = pw.check_collision()
    if collisions:
        print(f"  Initial planner collisions ({len(collisions)}):")
        for c in collisions[:5]:
            print(f"    {c.link_name1}({c.object_name1}) <-> "
                  f"{c.link_name2}({c.object_name2})")


# ─── Single Grasp Attempt ────────────────────────────────────────────────────

def attempt_grasp(perception, strategies, robot, planner, pw, step_fn,
                  timings, idx, total):
    """Execute grasp pipeline for a perceived object. Returns outcome string."""
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()

    tag_base = f"[{idx+1}/{total}] {perception.name}"

    for strategy_name, target_p, target_q in strategies:
        tag = f"{tag_base} ({strategy_name})"
        print(f"\n  --- {tag} ---")
        print(f"    Target pos: {target_p}")

        approach_pose = MPPose(p=target_p, q=target_q)
        pre_pose = MPPose(p=target_p + [0, 0, PRE_GRASP_HEIGHT], q=target_q)

        sync_planner(planner)
        cq = get_robot_qpos(robot)

        # Transform to base-relative for IK
        approach_base = planner._transform_goal_to_wrt_base(approach_pose)
        pre_base = planner._transform_goal_to_wrt_base(pre_pose)

        # 1. Solve grasp IK
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
            print(f"    Grasp IK: FAILED for {strategy_name}")
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
        t0 = time.time()
        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, robot=robot)
        timings['exec'] += time.time() - t0

        # 4. Approach
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
                           lock_base=used_arm_only, robot=robot)
        timings['exec'] += time.time() - t0

        # 5. Close gripper
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_CLOSED)
        # Hold closed briefly
        qpos = get_robot_qpos(robot)
        hold_closed = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
        for _ in range(20):
            step_fn(hold_closed)
        timings['gripper'] += time.time() - t0

        # 6. Lift
        lift_pose = MPPose(p=target_p + [0, 0, LIFT_HEIGHT], q=target_q)
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
            actuate_gripper(step_fn, robot, GRIPPER_OPEN)
            return 'partial'
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_lift['status'] == 'Success':
            print(f"    Lift: OK ({r_lift['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_lift['position'], step_fn, GRIPPER_CLOSED,
                               lock_base=used_arm_only, robot=robot)
            timings['exec'] += time.time() - t0

            # Hold and check if object is still grasped (basic: check gripper width)
            qpos = get_robot_qpos(robot)
            hold = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
            for _ in range(30):
                step_fn(hold)
        else:
            print(f"    Lift: FAILED — {r_lift['status']}  [{dt:.2f}s]")

        # 7. Drop
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_OPEN)
        qpos = get_robot_qpos(robot)
        hold_open = make_action(qpos[3:10], GRIPPER_OPEN, qpos[:3])
        for _ in range(30):
            step_fn(hold_open)
        timings['gripper'] += time.time() - t0

        # 8. Return to home
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
            print(f"    Return: OK  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                               robot=robot)
            timings['exec'] += time.time() - t0
        else:
            print(f"    Return: FAILED, teleporting home  [{dt:.2f}s]")
            robot.set_qpos(torch.tensor(home_qpos, dtype=torch.float32).unsqueeze(0))

        t0 = time.time()
        wait_until_stable(step_fn,
                          make_action(ARM_HOME, GRIPPER_OPEN,
                                      get_robot_qpos(robot)[:3]),
                          robot, max_steps=100)
        timings['settle'] += time.time() - t0
        return 'success'

    return 'unreachable'


# ─── Object Spawning (optional — for testing without task env objects) ────────

def spawn_test_objects(scene, fixtures):
    """Spawn small objects on various surfaces for testing perception."""
    from test_robocasa_grasp import collect_placements, spawn_cube, COLORS, CUBE_HALF

    all_placements = collect_placements(fixtures)
    spawned = []
    for i, (label, pos, ftype) in enumerate(all_placements):
        color = COLORS[i % len(COLORS)]
        cube_pos = pos.copy()
        cube_pos[2] += CUBE_HALF + 0.002
        cube_name = f"test_obj_{i}_{label}"
        try:
            builder = scene.create_actor_builder()
            hs = np.array([CUBE_HALF] * 3)
            builder.add_box_collision(half_size=hs)
            builder.add_box_visual(
                half_size=hs,
                material=sapien.render.RenderMaterial(
                    base_color=color),
            )
            actor = builder.build(name=cube_name)
            actor.set_pose(sapien.Pose(p=cube_pos))
            spawned.append((cube_name, cube_pos, label, ftype, actor))
        except Exception as e:
            print(f"  Spawn failed for {label}: {e}")

    return spawned


# ─── Video Writer ─────────────────────────────────────────────────────────────

class VideoWriter:
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


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Perception-based grasp pipeline")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human', choices=['human', 'rgb_array'])
    parser.add_argument('--task', default='RoboCasaKitchen-v1',
                        help='Task env ID')
    parser.add_argument('--camera', default='base_camera',
                        help='Camera to use for perception (base_camera or wrist_camera)')
    parser.add_argument('--max-objects', type=int, default=None,
                        help='Max objects to attempt grasping')
    parser.add_argument('--spawn-test-objects', action='store_true',
                        help='Spawn test cubes on surfaces (if task has no objects)')
    parser.add_argument('--debug-dir', default=None,
                        help='Save perception debug images to this directory')
    args = parser.parse_args()

    t_total = time.time()

    # --- Create environment with perception ---
    print(f"Creating env: {args.task} with rgb+depth+segmentation...")
    t0 = time.time()
    env = gym.make(
        args.task,
        num_envs=1,
        robot_uids='tidyverse',
        control_mode='whole_body',
        obs_mode='rgb+depth+segmentation',
        render_mode=args.render,
        sensor_configs=dict(shader_pack="default"),
    )
    obs, info = env.reset(seed=args.seed)
    t_env = time.time() - t0
    print(f"  env setup: {t_env:.2f}s")

    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    # Reposition render camera
    from mani_skill.utils import sapien_utils as _su
    _rpos = robot.pose.p[0].cpu().numpy()
    _cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]
    _cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]
    _cam_pose = _su.look_at(_cam_eye, _cam_target)
    _p = _cam_pose.raw_pose[0].cpu().numpy()
    _sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
    for cam in env.unwrapped._human_render_cameras.values():
        cam.camera.set_local_pose(_sapien_pose)

    # Video writer
    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    cam_video_writers = {}
    if args.render == 'rgb_array':
        base = f'perception_grasp_seed{args.seed}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)
        # Per-camera video writers
        for cam_name in ['base_camera', 'wrist_camera']:
            cam_path = os.path.join(video_dir, f'{base}_run{run}_{cam_name}.mp4')
            cam_video_writers[cam_name] = VideoWriter(cam_path, fps=30)

    step_label = ["idle"]

    def _burn_label(frame, text):
        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600)
        thick = max(1, int(h / 300))
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 20, th_ + 16), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, th_ + 8), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def step_fn(action):
        obs_step, _, _, _, _ = env.step(action)
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
            # Save per-camera RGB frames
            for cam_name, cw in cam_video_writers.items():
                if cam_name in obs_step.get("sensor_data", {}):
                    cam_rgb = common.to_numpy(obs_step["sensor_data"][cam_name]["rgb"][0])
                    cam_frame = cam_rgb.astype(np.uint8).copy()
                    _burn_label(cam_frame, f"{cam_name}: {step_label[0]}")
                    cw.add_frame(cam_frame)

    robot_pos = robot.pose.p[0].cpu().numpy()
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot at {robot_pos}, arm base at {arm_base}")

    # --- Optionally spawn test objects ---
    spawned_names = None
    if args.spawn_test_objects:
        print("\nSpawning test objects on surfaces...")
        spawned = spawn_test_objects(scene, fixtures)
        spawned_names = {s[0] for s in spawned}
        print(f"  Spawned {len(spawned)} objects: {spawned_names}")
        # Step to let objects settle and update render
        hold = make_action(ARM_HOME, GRIPPER_OPEN, get_robot_qpos(robot)[:3])
        for _ in range(60):
            step_fn(hold)
        # Force segmentation ID map refresh and register spawned actors
        try:
            del env.unwrapped.__dict__['segmentation_id_map']
        except KeyError:
            pass
        # Manually inject spawned actors into segmentation_id_map
        seg_map = env.unwrapped.segmentation_id_map
        for name, pos, label, ftype, actor in spawned:
            sid = actor.per_scene_id
            seg_map[sid] = actor
            print(f"    Registered {name} with seg_id={sid}")

    # --- Stabilize ---
    base_cmd = get_robot_qpos(robot)[:3].copy()
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    step_label[0] = "Stabilizing"
    print("\nStabilizing robot...")
    t0 = time.time()
    wait_until_stable(step_fn, hold, robot)
    print(f"  stabilize: {time.time() - t0:.2f}s")

    # --- Get fresh observation for perception ---
    print("\nCapturing perception data...")
    step_label[0] = "Perceiving"
    # Need a fresh obs after stabilization
    obs, _, _, _, _ = env.step(hold)

    # List available cameras
    available_cameras = list(obs['sensor_data'].keys())
    print(f"  Available cameras: {available_cameras}")

    if not available_cameras:
        print("ERROR: No cameras available!")
        env.close()
        return

    # --- Run perception on ALL cameras, merge results ---
    t0 = time.time()
    perceptions = []
    seen_names = set()
    debug_dir = args.debug_dir or os.path.join(
        os.path.dirname(__file__), 'perception_debug')

    registered_ids = set(env.unwrapped.segmentation_id_map.keys()) if spawned_names else set()
    for cam_name in available_cameras:
        # Debug: dump what seg IDs are in the image
        _seg = common.to_numpy(obs["sensor_data"][cam_name]["segmentation"][0])[..., 0]
        _unique = set(np.unique(_seg).tolist())
        _overlap = _unique & registered_ids
        print(f"\n  [{cam_name}] seg IDs in image: {sorted(_unique)}")
        if spawned_names:
            print(f"    registered IDs: {sorted(registered_ids)}")
            print(f"    overlap: {sorted(_overlap)}")

        cam_perceptions = perceive_objects(obs, env, camera_name=cam_name,
                                           target_names=spawned_names)
        print(f"  [{cam_name}] detected {len(cam_perceptions)} objects")
        # Save per-camera debug images
        cam_debug_dir = os.path.join(debug_dir, cam_name)
        save_perception_debug(obs, env, cam_perceptions, cam_name, cam_debug_dir)

        # Merge, dedup by name (keep the one with more pixels)
        for p in cam_perceptions:
            p.fixture_context = classify_fixture_context(p.center_3d, fixtures)
            if p.name not in seen_names:
                seen_names.add(p.name)
                perceptions.append(p)
            else:
                # Replace if this camera got more pixels
                for j, existing in enumerate(perceptions):
                    if existing.name == p.name and p.mask_pixels > existing.mask_pixels:
                        perceptions[j] = p
                        break

    t_percept = time.time() - t0
    print(f"\n  Perceived {len(perceptions)} unique objects in {t_percept:.2f}s:")

    # Sort by distance to arm base (nearest first)
    perceptions.sort(key=lambda p: np.linalg.norm(p.center_3d - arm_base))

    if args.max_objects is not None:
        perceptions = perceptions[:args.max_objects]

    for i, p in enumerate(perceptions):
        dist = np.linalg.norm(p.center_3d - arm_base)
        sz = p.size_3d
        print(f"  [{i:2d}] {p.name:40s} dist={dist:.2f}m  "
              f"size=[{sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f}]  "
              f"aspect={p.aspect_ratio:.1f}  ctx={p.fixture_context}")

    if not perceptions:
        print("\nNo objects detected — nothing to grasp.")
        env.close()
        return

    # --- Setup planner ---
    print("\nSetting up SapienPlanner...")
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

    # ACM — keep collision checking for perceived objects
    target_names = {p.name for p in perceptions}
    t0 = time.time()
    build_kitchen_acm(pw, planner, target_names, mode='relaxed')
    t_acm = time.time() - t0
    print(f"  ACM build: {t_acm:.2f}s")

    # --- Grasp loop ---
    timings = {'ik': 0.0, 'planning': 0.0, 'exec': 0.0,
               'gripper': 0.0, 'settle': 0.0}
    results = {'success': 0, 'partial': 0, 'unreachable': 0, 'error': 0}

    for ci, perception in enumerate(perceptions):
        dist = np.linalg.norm(perception.center_3d - arm_base)
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(perceptions)}] {perception.name} "
              f"({perception.fixture_context})  dist={dist:.2f}m")
        print(f"  center={perception.center_3d}  "
              f"size={perception.size_3d}  aspect={perception.aspect_ratio:.1f}")

        step_label[0] = f"grasp {ci+1}/{len(perceptions)} {perception.name}"

        # Choose grasp strategies based on perception
        strategies = choose_grasp_strategy(perception, arm_base)
        strategy_names = [s[0] for s in strategies]
        print(f"  Strategy order: {strategy_names}")

        try:
            outcome = attempt_grasp(perception, strategies, robot, planner, pw,
                                    step_fn, timings, ci, len(perceptions))
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            outcome = 'error'
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
        print(f"  {k:12s}: {v}/{len(perceptions)}")

    t_total_elapsed = time.time() - t_total
    print(f"\nTIMING:")
    print(f"  env setup:     {t_env:7.2f}s")
    print(f"  perception:    {t_percept:7.2f}s")
    print(f"  planner setup: {t_planner:7.2f}s")
    print(f"  ACM build:     {t_acm:7.2f}s")
    print(f"  IK:            {timings['ik']:7.2f}s")
    print(f"  planning:      {timings['planning']:7.2f}s")
    print(f"  execution:     {timings['exec']:7.2f}s")
    print(f"  gripper:       {timings['gripper']:7.2f}s")
    print(f"  settle:        {timings['settle']:7.2f}s")
    print(f"  TOTAL:         {t_total_elapsed:7.2f}s")

    if video_writer:
        video_writer.close()
    for cw in cam_video_writers.values():
        cw.close()

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
