"""Perception pipeline: extract object detections from depth + segmentation cameras."""
import os
import numpy as np
import cv2

from mani_skill.utils import common
from mani_skill.utils.structs import Link

# Fixture types for context classification
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mani_skill.utils.scene_builder.robocasa.fixtures.stove import Stove, Stovetop
from mani_skill.utils.scene_builder.robocasa.fixtures.cabinet import (
    SingleCabinet, HingeCabinet, OpenCabinet, Drawer,
)
from mani_skill.utils.scene_builder.robocasa.fixtures.microwave import Microwave
from mani_skill.utils.scene_builder.robocasa.fixtures.others import Floor, Wall


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

    # Get depth at each pixel (mm -> m)
    depths_mm = depth_img[pixels_uv[:, 1], pixels_uv[:, 0]].astype(float)
    depths_m = depths_mm / 1000.0

    # Back-project to camera frame (OpenCV convention: x right, y down, z forward)
    x_cam = (u - cx) * depths_m / fx
    y_cam = (v - cy) * depths_m / fy
    z_cam = depths_m

    # Convert OpenCV camera frame to OpenGL camera frame
    # OpenGL: x right, y up, z backward -> flip y and z
    x_gl = x_cam
    y_gl = -y_cam
    z_gl = -z_cam

    pts_gl = np.stack([x_gl, y_gl, z_gl, np.ones_like(z_gl)], axis=-1)  # (N, 4)

    # Transform to world using cam2world_gl (4x4)
    pts_world = (cam2world_gl @ pts_gl.T).T[:, :3]  # (N, 3)

    return pts_world


def _build_config_name_to_category(env):
    """Build mapping from config names (e.g. 'obj') to category names (e.g. 'mug_0').

    Uses object_cfgs which stores the sampled object info including the real
    category. Deduplicates by appending _0, _1, etc. for repeated categories.

    Returns:
        dict: config_name -> semantic_name (e.g. {'obj': 'mug_0', 'distr_counter': 'banana_0'})
    """
    scene_idx = getattr(env.unwrapped, '_scene_idx_to_be_loaded', 0)
    object_cfgs = getattr(env.unwrapped, 'object_cfgs', None)
    if object_cfgs is None or scene_idx >= len(object_cfgs):
        return {}

    # First pass: collect categories to know which need dedup suffixes
    cat_counts = {}
    entries = []
    for cfg in object_cfgs[scene_idx]:
        info = cfg.get("info")
        if info and "cat" in info:
            cat = info["cat"]
            idx = cat_counts.get(cat, 0)
            cat_counts[cat] = idx + 1
            entries.append((cfg["name"], cat, idx))
        else:
            entries.append((cfg["name"], None, 0))

    # Second pass: only add _N suffix if category appears more than once
    mapping = {}
    for config_name, cat, idx in entries:
        if cat is not None:
            semantic = f"{cat}_{idx}" if cat_counts[cat] > 1 else cat
            mapping[config_name] = semantic
            # MJCFObject.build() appends _{scene_idx} to actor names,
            # so "obj" becomes "obj_0" in scene 0. Map both forms.
            mapping[f"{config_name}_{scene_idx}"] = semantic
    return mapping


def perceive_objects(obs, env, camera_name="base_camera",
                     min_pixels=50, max_depth_mm=5000,
                     target_names=None, skip_filter=True):
    """Extract object detections from camera observations.

    Args:
        obs: observation dict from env.step() or env.reset()
        env: the unwrapped ManiSkill env
        camera_name: which camera to use
        min_pixels: minimum mask area to consider
        max_depth_mm: max depth to consider (ignore far objects)
        target_names: if set, only detect these object names (skip all others).
            Matches against semantic category names (e.g. 'mug', 'mug_0').

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

    # Build config name -> semantic category mapping
    name_to_cat = _build_config_name_to_category(env)

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

        raw_name = obj.name if hasattr(obj, 'name') else str(obj)
        is_robot = raw_name in robot_link_names
        # Resolve to semantic category name if available
        obj_name = name_to_cat.get(raw_name, raw_name)

        # Skip robot links
        if is_robot:
            continue

        # Filter by target names or skip known fixture prefixes
        if target_names is not None:
            if obj_name not in target_names:
                continue
        elif skip_filter:
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

        bbox_min = np.min(pts_3d, axis=0)
        bbox_max = np.max(pts_3d, axis=0)
        center_3d = (bbox_min + bbox_max) / 2.0  # bbox midpoint, not surface mean

        # Fit ellipse to the 2D mask for shape analysis
        ellipse_axes = (0.0, 0.0)
        ellipse_angle = 0.0
        if len(valid_pixels) >= 5:
            try:
                ellipse = cv2.fitEllipse(valid_pixels.astype(np.float32))
                avg_depth_m = np.mean(valid_depths) / 1000.0
                fx = intrinsic[0, 0]
                fy = intrinsic[1, 1]
                major_m = ellipse[1][1] * avg_depth_m / fy
                minor_m = ellipse[1][0] * avg_depth_m / fx
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


def perceive_by_seg_id(obs, seg_id, camera_name="base_camera", min_pixels=20,
                       max_depth_mm=5000):
    """Back-project a known seg_id from camera data to 3D world position.

    Unlike perceive_objects which discovers objects, this targets a specific
    segmentation ID (e.g., a drawer link whose seg_id is already known).

    Returns PerceptionResult or None if not visible.
    """
    sensor_data = obs["sensor_data"][camera_name]
    depth = common.to_numpy(sensor_data["depth"][0])[..., 0]
    seg = common.to_numpy(sensor_data["segmentation"][0])[..., 0]
    sensor_params = obs["sensor_param"][camera_name]
    intrinsic = common.to_numpy(sensor_params["intrinsic_cv"][0])
    cam2world = common.to_numpy(sensor_params["cam2world_gl"][0])

    mask = seg == seg_id
    n_pixels = int(mask.sum())
    if n_pixels < min_pixels:
        return None

    ys, xs = np.where(mask)
    pixels = np.stack([xs, ys], axis=-1)
    pixel_depths = depth[ys, xs]
    valid = (pixel_depths > 0) & (pixel_depths < max_depth_mm)
    if valid.sum() < min_pixels // 2:
        return None

    valid_pixels = pixels[valid]
    pts_3d = deproject_pixels_to_world(valid_pixels, depth, intrinsic, cam2world)
    finite_mask = np.all(np.isfinite(pts_3d), axis=1)
    pts_3d = pts_3d[finite_mask]
    if len(pts_3d) < 5:
        return None

    bbox_min = np.min(pts_3d, axis=0)
    bbox_max = np.max(pts_3d, axis=0)
    center_3d = (bbox_min + bbox_max) / 2.0

    # Fit plane to 3D points via PCA to get surface normal
    surface_normal = None
    if len(pts_3d) >= 10:
        centroid = np.mean(pts_3d, axis=0)
        centered = pts_3d - centroid
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        normal = vh[2]  # smallest singular value = plane normal
        # Orient normal toward the camera (away from surface)
        cam_pos = cam2world[:3, 3]
        if np.dot(normal, cam_pos - centroid) < 0:
            normal = -normal
        surface_normal = normal

    result = PerceptionResult(
        name=f"seg_{seg_id}",
        seg_id=seg_id,
        center_3d=center_3d,
        bbox_3d_min=bbox_min,
        bbox_3d_max=bbox_max,
        ellipse_axes=(0.0, 0.0),
        ellipse_angle=0.0,
        mask_pixels=n_pixels,
    )
    result.surface_normal = surface_normal
    return result


def find_handle_targets(fixtures, env, fixture_types=None):
    """Discover handle link names from articulated fixtures for perception targeting.

    Args:
        fixtures: dict of fixture_name -> fixture object
        env: ManiSkill env (unwrapped)
        fixture_types: tuple of fixture classes to include (default: Drawer only)

    Returns:
        list of dicts with keys: fixture_name, fixture_type, link_name, link,
            articulation, fixture, front_dir, fixture_yaw
    """
    if fixture_types is None:
        fixture_types = (Drawer,)

    seg_map = env.segmentation_id_map
    targets = []
    for fname, fix in fixtures.items():
        if not isinstance(fix, fixture_types):
            continue
        if not getattr(fix, 'is_articulation', False):
            continue
        art = fix.articulation
        if art is None:
            continue

        fxt_yaw = fix.euler[2] if hasattr(fix, 'euler') and len(fix.euler) > 2 else 0.0
        front_dir = np.array([np.sin(fxt_yaw), -np.cos(fxt_yaw), 0.0])

        for link in art.get_links():
            lname = link.get_name()
            if lname in ('dummy_root_0', 'object'):
                continue
            if 'knob' in lname.lower():
                continue

            # Register this link in segmentation map so perceive_objects can find it
            sid = int(link.per_scene_id)
            seg_map[sid] = link

            targets.append({
                'fixture_name': fname,
                'fixture_type': type(fix).__name__,
                'link_name': lname,
                'link': link,
                'articulation': art,
                'fixture': fix,
                'front_dir': front_dir,
                'fixture_yaw': fxt_yaw,
            })

    return targets


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

        if isinstance(fix, (SingleCabinet, HingeCabinet, OpenCabinet, Drawer, Microwave)):
            if hasattr(fix, 'get_int_sites') and hasattr(fix, '_bounds_sites'):
                try:
                    if 'int_p0' in fix._bounds_sites:
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
        obj_mask = seg == p.seg_id
        ys, xs = np.where(obj_mask)
        if len(xs) < 5:
            continue

        x1, y1 = xs.min(), ys.min()
        x2, y2 = xs.max(), ys.max()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        try:
            pts = np.stack([xs, ys], axis=-1).astype(np.float32)
            ellipse = cv2.fitEllipse(pts)
            cv2.ellipse(annotated, ellipse, (255, 0, 0), 2)
        except cv2.error:
            pass

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
