"""Grasp strategy selection: choose grasp poses based on perceived object properties."""
import numpy as np
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R


# ---------------------------------------------------------------------------
# Shared helpers used by both perception-grasp and robocasa-grasp pipelines
# ---------------------------------------------------------------------------

DROP_HEIGHT = 0.05


def build_object_grasps(obj_pos, arm_base, obj_yaw=0.0):
    """Grasp poses for free objects (blocks, cups, etc.): Angled45 + Top-Down.

    Generates 4 yaw rotations (0, 90, 180, 270 deg) relative to the
    object's own yaw so gripper fingers align with object faces.
    """
    grasps = []
    for i, yaw_offset in enumerate([0, np.pi / 2, np.pi, -np.pi / 2]):
        yaw = obj_yaw + yaw_offset
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        deg = int(np.degrees(yaw_offset))
        grasps.append((
            f'Angled45-{deg}',
            obj_pos + np.array([-0.02 * cos_y, -0.02 * sin_y, 0.02]),
            np.array(euler2quat(0, 3 * np.pi / 4, yaw)),
        ))
    for i, yaw_offset in enumerate([0, np.pi / 2, np.pi, -np.pi / 2]):
        yaw = obj_yaw + yaw_offset
        deg = int(np.degrees(yaw_offset))
        grasps.append((
            f'Top-Down-{deg}',
            obj_pos.copy(),
            np.array(euler2quat(0, np.pi, yaw)),
        ))
    return grasps


def build_handle_grasps(handle_pos, arm_base):
    """Grasp poses for handles: Front + Front-Vertical (rotated 90 deg)."""
    yaw = np.arctan2(handle_pos[1] - arm_base[1],
                     handle_pos[0] - arm_base[0])
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    front_rot = R.from_euler('yz', [np.pi / 2, yaw])
    front_vert_rot = front_rot * R.from_euler('z', np.pi / 2)
    return [
        ('Front',
         handle_pos + np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08]),
         np.array(front_rot.as_quat()[[3, 0, 1, 2]])),
        ('Front-Vertical',
         handle_pos + np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08]),
         np.array(front_vert_rot.as_quat()[[3, 0, 1, 2]])),
    ]


def select_grasps(obj_pos, arm_base, ftype, label, obj_yaw=0.0):
    """Select ordered grasp list based on object/fixture type."""
    is_handle = 'handle' in label.lower()
    if is_handle:
        return build_handle_grasps(obj_pos, arm_base)
    grasps = build_object_grasps(obj_pos, arm_base, obj_yaw=obj_yaw)
    is_enclosed = 'interior' in label
    if is_enclosed:
        return list(reversed(grasps))
    elif ftype in ('Stove', 'Stovetop'):
        return [g for g in grasps if g[0].startswith('Top-Down')] + \
               [g for g in grasps if not g[0].startswith('Top-Down')]
    return grasps


def build_place_poses(dest_pos, arm_base):
    """Place poses: Angled45 (preferred) then Top-Down, 4 world-frame yaws each."""
    poses = []
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        deg = int(np.degrees(yaw))
        p = dest_pos.copy()
        p[2] += DROP_HEIGHT
        p += np.array([-0.02 * cos_y, -0.02 * sin_y, 0.0])
        q = np.array(euler2quat(0, 3 * np.pi / 4, yaw))
        poses.append((f'Place-Angled45-{deg}', p, q))
    for yaw in [0, np.pi / 2, np.pi, -np.pi / 2]:
        deg = int(np.degrees(yaw))
        p = dest_pos.copy()
        p[2] += DROP_HEIGHT
        q = np.array(euler2quat(0, np.pi, yaw))
        poses.append((f'Place-TopDown-{deg}', p, q))
    return poses


# ---------------------------------------------------------------------------
# Perception-grasp specific: strategy selection with yaw variations
# ---------------------------------------------------------------------------

def _build_perception_grasp_pose(strategy, obj_pos, yaw):
    """Build a single (strategy_name, pos, quat) for a given yaw."""
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    if strategy == 'TopDown':
        return (f'TopDown@{np.degrees(yaw):.0f}',
                obj_pos.copy(),
                np.array([0, 1, 0, 0], dtype=float))
    elif strategy == 'Angled45':
        offset = np.array([-0.02 * cos_y, -0.02 * sin_y, 0.02])
        q = np.array(euler2quat(0, 3 * np.pi / 4, yaw), dtype=float)
        return (f'Angled45@{np.degrees(yaw):.0f}',
                obj_pos + offset, q)
    # # Front: gripper horizontal, approaching from robot's direction
    # elif strategy == 'Front':
    #     front_offset = np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08])
    #     front_rot = R.from_euler('yz', [np.pi / 2, yaw])
    #     front_q = front_rot.as_quat()[[3, 0, 1, 2]]  # xyzw -> wxyz
    #     return (f'Front@{np.degrees(yaw):.0f}',
    #             obj_pos + front_offset, front_q.astype(float))
    # # Front 90: rotated 90 from front (for handles perpendicular to approach)
    # elif strategy == 'Front90':
    #     front_offset = np.array([-0.06 * cos_y, -0.06 * sin_y, 0.08])
    #     front90_rot = R.from_euler('xyz', [np.pi / 2, 0, yaw])
    #     front90_q = front90_rot.as_quat()[[3, 0, 1, 2]]
    #     return (f'Front90@{np.degrees(yaw):.0f}',
    #             obj_pos + front_offset, front90_q.astype(float))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


YAW_OFFSETS = [0.0, np.radians(30), np.radians(-30)]


def choose_grasp_strategy(perception, arm_base_pos):
    """Choose grasp strategy based on perceived object shape and context.

    Returns list of (strategy_name, grasp_pose_p, grasp_pose_q) tuples,
    ordered by preference. Each strategy is tried at 3 yaw angles:
    direct (arm->object), +30, and -30.

    Active strategies:
        - TopDown:   gripper pointing straight down (z = [0,0,-1])
        - Angled45:  gripper at 45 tilt toward object
    """
    obj_pos = perception.center_3d
    ctx = perception.fixture_context

    # Base yaw: direction from arm base to object
    base_yaw = np.arctan2(obj_pos[1] - arm_base_pos[1],
                          obj_pos[0] - arm_base_pos[0])

    # Strategy order based on context
    if ctx in ('cabinet_interior', 'drawer_interior', 'microwave_interior'):
        order = ['Angled45', 'TopDown']
    elif ctx == 'stove':
        order = ['TopDown', 'Angled45']
    else:
        order = ['TopDown', 'Angled45']

    # Build candidates: each strategy x each yaw offset
    ordered = []
    for strategy in order:
        for yaw_off in YAW_OFFSETS:
            yaw = base_yaw + yaw_off
            ordered.append(_build_perception_grasp_pose(strategy, obj_pos, yaw))

    return ordered
