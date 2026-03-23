"""Motion planning utilities: monkey-patch, AABB computation, fixture boxes, ACM builder."""
import numpy as np
import sapien

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanningWorld
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
import mplib.sapien_utils.conversion as _conv

from mani_skill.utils.scene_builder.robocasa.fixtures.others import Floor, Wall


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


# ─── AABB Computation ────────────────────────────────────────────────────────

def _shape_aabb_corners(shape, shape_T):
    """Get world-frame corners/extremes of a collision shape for AABB computation."""
    pts = []
    if isinstance(shape, PhysxCollisionShapeBox):
        hs = np.array(shape.half_size)
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    local = np.array([sx * hs[0], sy * hs[1], sz * hs[2]])
                    pts.append(shape_T[:3, :3] @ local + shape_T[:3, 3])
    elif isinstance(shape, PhysxCollisionShapeConvexMesh):
        verts = shape.vertices
        if not np.allclose(shape.scale, 1.0):
            verts = verts * np.array(shape.scale)
        pts_w = (shape_T[:3, :3] @ verts.T).T + shape_T[:3, 3]
        pts.extend(pts_w)
    elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
        verts = shape.vertices
        pts_w = (shape_T[:3, :3] @ verts.T).T + shape_T[:3, 3]
        pts.extend(pts_w)
    elif isinstance(shape, PhysxCollisionShapeSphere):
        c = shape_T[:3, 3]
        r = shape.radius
        for axis in range(3):
            for sign in [-1, 1]:
                p = c.copy()
                p[axis] += sign * r
                pts.append(p)
    elif isinstance(shape, (PhysxCollisionShapeCapsule, PhysxCollisionShapeCylinder)):
        c = shape_T[:3, 3]
        r = shape.radius
        hl = shape.half_length
        local_axis = shape_T[:3, 0]
        for end in [-hl, hl]:
            center = c + local_axis * end
            for ax in range(3):
                for sign in [-1, 1]:
                    p = center.copy()
                    p[ax] += sign * r
                    pts.append(p)
    return pts


def compute_articulation_aabb(scene_art):
    """Compute world-frame AABB of a SAPIEN articulation from collision shapes."""
    all_pts = []
    for link in scene_art.get_links():
        link_T = link.pose.to_transformation_matrix()
        for shape in link.collision_shapes:
            local_T = shape.local_pose.to_transformation_matrix()
            shape_T = link_T @ local_T
            pts = _shape_aabb_corners(shape, shape_T)
            all_pts.extend(pts)
    if not all_pts:
        return None, None
    all_pts = np.array(all_pts)
    return all_pts.min(axis=0), all_pts.max(axis=0)


def _compute_fixture_aabb(scene, fname):
    """Compute world-frame AABB for a fixture by checking articulations then static actors."""
    scene_arts = {a.name: a for a in scene.get_all_articulations()}

    # Try articulations first
    for art_name, art in scene_arts.items():
        if fname.replace(' ', '_') in art_name or fname in art_name:
            return compute_articulation_aabb(art)

    # Try fuzzy match on articulations
    fname_parts = fname.replace(' ', '_').split('_')
    for art_name, art in scene_arts.items():
        art_lower = art_name.lower()
        if all(part.lower() in art_lower for part in fname_parts if len(part) > 2):
            return compute_articulation_aabb(art)

    # Try static actors
    for actor in scene.get_all_actors():
        if fname.replace(' ', '_') in actor.name or fname in actor.name:
            all_pts = []
            actor_T = actor.pose.to_transformation_matrix()
            comp = actor.find_component_by_type(
                sapien.physx.PhysxRigidStaticComponent
            ) or actor.find_component_by_type(
                sapien.physx.PhysxRigidDynamicComponent
            )
            if comp is None:
                return None, None
            for shape in comp.collision_shapes:
                local_T = shape.local_pose.to_transformation_matrix()
                shape_T = actor_T @ local_T
                pts = _shape_aabb_corners(shape, shape_T)
                all_pts.extend(pts)
            if all_pts:
                all_pts_arr = np.array(all_pts)
                return all_pts_arr.min(axis=0), all_pts_arr.max(axis=0)
            return None, None

    return None, None


# ─── Fixture Boxes ───────────────────────────────────────────────────────────

def add_fixture_boxes_to_planner(pw, scene, fixtures_dict, skip_fixtures=None):
    """Add AABB box approximations of fixtures directly to the mplib planning world.

    These boxes are added as FCL objects to the planning world only — they do NOT
    affect SAPIEN physics. The original fixture meshes should be relaxed in the ACM.

    skip_fixtures: set of fixture names to skip (e.g. sink for drop target)
    """
    skip_fixtures = skip_fixtures or set()
    box_names = []
    for fname, fix in fixtures_dict.items():
        if isinstance(fix, (Floor, Wall)):
            continue
        if not hasattr(fix, 'pos'):
            continue
        if fname in skip_fixtures:
            continue

        bbox_min, bbox_max = _compute_fixture_aabb(scene, fname)

        # Fallback only: use fixture pos+size when no collision AABB found
        if bbox_min is None and hasattr(fix, 'size') and fix.size is not None:
            pos = np.array(fix.pos)
            size = np.array(fix.size)
            if len(size) == 3 and np.all(size > 0.01):
                fix_half = size / 2.0
                bbox_min = pos - fix_half
                bbox_max = pos + fix_half

        if bbox_min is None:
            continue

        center = (bbox_min + bbox_max) / 2.0
        half_size = (bbox_max - bbox_min) / 2.0

        # Skip tiny or degenerate boxes
        if np.any(half_size < 0.005) or np.any(half_size > 5.0):
            continue

        box_name = f"fixture_box_{fname}"
        try:
            box_geom = Box(side=half_size * 2)
            shape = CollisionObject(box_geom)
            fcl_obj = FCLObject(box_name, MPPose(p=center), [shape], [MPPose()])
            pw.add_object(fcl_obj)
            box_names.append(box_name)
        except Exception as e:
            print(f"    Failed to add box for {fname}: {e}")

    return box_names


# ─── ACM Builder ─────────────────────────────────────────────────────────────

def _get_object_position(pw, name):
    """Get the world position of a planning-world object."""
    try:
        return np.array(pw.get_object(name).pose.p)
    except Exception:
        return None


def build_kitchen_acm(pw, planner, target_names=None, mode='relaxed',
                      robot_pos=None, target_positions=None, near_radius=1.5):
    """Configure ACM for collision checking.

    mode='relaxed': relax ALL fixture collisions (planner ignores furniture).
    mode='strict':  only relax fixtures far from both robot AND all target
                    positions (>near_radius); nearby fixtures are checked.

    target_positions: list of [x,y,z] positions the robot will travel to
                      (e.g. grasp targets, sink). Objects near any of these
                      are kept as collision obstacles.
    """
    acm = pw.get_allowed_collision_matrix()
    art_names = pw.get_articulation_names()
    robot_link_names = planner.pinocchio_model.get_link_names()
    robot_art = next(n for n in art_names if 'tidyverse' in n.lower())

    target_set = set(target_names or [])

    # --- Log what the planner sees ---
    print(f"\n  Planning world contents (ACM mode={mode}):")
    print(f"    Robot links ({len(robot_link_names)}): {robot_link_names[:5]}...")
    print(f"    Articulations ({len(art_names)}):")

    for an in art_names:
        if an == robot_art:
            print(f"      [ROBOT] {an}")
            continue
        fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        if mode == 'relaxed':
            print(f"      [RELAXED] {an} ({len(fl)} links)")
        else:
            print(f"      [PENDING] {an} ({len(fl)} links)")

    obj_names = pw.get_object_names()
    checked_objs = [n for n in obj_names if n in target_set]

    # Build list of reference points: robot start + all target positions
    ref_points = []
    if robot_pos is not None:
        ref_points.append(np.array(robot_pos)[:2])
    for tp in (target_positions or []):
        ref_points.append(np.array(tp)[:2])

    # Classify static objects
    relaxed_static, checked_static = [], []
    for on in obj_names:
        if on in target_set:
            continue
        if mode == 'relaxed':
            relaxed_static.append(on)
        else:
            pos = _get_object_position(pw, on)
            if pos is not None and ref_points:
                # Near if within near_radius of ANY reference point
                min_dist = min(np.linalg.norm(pos[:2] - rp) for rp in ref_points)
                if min_dist > near_radius:
                    relaxed_static.append(on)
                else:
                    checked_static.append((on, min_dist))
            else:
                checked_static.append((on, -1))

    print(f"    Static objects ({len(obj_names)} total):")
    print(f"      Targets (always checked): {len(checked_objs)}")
    if mode == 'relaxed':
        print(f"      ACM-relaxed: {len(relaxed_static)}")
    else:
        print(f"      Collision-checked (near, <{near_radius}m): {len(checked_static)}")
        for on, d in sorted(checked_static, key=lambda x: x[1]):
            print(f"        - {on}  dist={d:.2f}m")
        print(f"      ACM-relaxed (far): {len(relaxed_static)}")

    # --- Apply ACM ---
    if mode == 'relaxed':
        for an in art_names:
            if an == robot_art:
                continue
            fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
            for rl in robot_link_names:
                for f in fl:
                    acm.set_entry(rl, f, True)
        for on in relaxed_static:
            for rl in robot_link_names:
                acm.set_entry(rl, on, True)
    else:
        # Strict: relax fixture articulations (too many false collisions
        # from articulated fixture meshes), but keep nearby static objects
        for an in art_names:
            if an == robot_art:
                continue
            fl = pw.get_articulation(an).get_pinocchio_model().get_link_names()
            for rl in robot_link_names:
                for f in fl:
                    acm.set_entry(rl, f, True)
        for on in relaxed_static:
            for rl in robot_link_names:
                acm.set_entry(rl, on, True)

    # Check initial collisions
    sync_planner(planner)
    collisions = pw.check_collision()
    if collisions:
        print(f"\n  Initial planner collisions ({len(collisions)}):")
        for c in collisions:
            print(f"    {c.link_name1}({c.object_name1}) <-> "
                  f"{c.link_name2}({c.object_name2})")
    else:
        print("\n  No initial collisions.")


# ─── Planner Sync ────────────────────────────────────────────────────────────

def sync_planner(planner):
    try:
        planner.update_from_simulation()
    except Exception:
        pass


# ─── Start-state Collision Resolution ────────────────────────────────────────

def check_start_collisions(pw, planner, target_name=None):
    """Check for start-state collisions, ignoring the target object.

    Returns list of (robot_link, obstacle_name) pairs for non-target collisions.
    """
    sync_planner(planner)
    collisions = pw.check_collision()
    if not collisions:
        return []

    robot_link_names = set(planner.pinocchio_model.get_link_names())
    problems = []
    for c in collisions:
        # Identify which side is robot, which is obstacle
        if c.object_name1 in robot_link_names:
            robot_link, obstacle = c.link_name1, c.object_name2
        elif c.object_name2 in robot_link_names:
            robot_link, obstacle = c.link_name2, c.object_name1
        else:
            continue  # self-collision or non-robot — skip

        # Skip collisions with the target object we're trying to grasp
        if target_name and target_name in obstacle:
            continue

        problems.append((robot_link, obstacle))

    return problems


def resolve_start_collisions(pw, planner, robot, step_fn, target_name=None,
                             max_attempts=5, nudge_dist=0.05):
    """Move the base to escape start-state collisions (excluding target object).

    Strategy: compute direction from colliding obstacle to robot base, nudge
    the base along that direction. Repeat up to max_attempts times.

    Returns True if collision-free (or only target collisions remain).
    """
    from execution import make_action, get_robot_qpos, ARM_HOME, GRIPPER_OPEN

    for attempt in range(max_attempts):
        problems = check_start_collisions(pw, planner, target_name)
        if not problems:
            return True

        robot_links, obstacles = zip(*problems)
        unique_obstacles = set(obstacles)
        print(f"    Start-state collision (attempt {attempt+1}/{max_attempts}): "
              f"robot <-> {unique_obstacles}")

        # Get obstacle positions to compute nudge direction
        robot_pos = robot.pose.p[0].cpu().numpy()[:2]  # xy only
        nudge_dir = np.zeros(2)

        for obs_name in unique_obstacles:
            obs_pos = _get_object_position(pw, obs_name)
            if obs_pos is not None:
                away = robot_pos - obs_pos[:2]
                norm = np.linalg.norm(away)
                if norm > 1e-3:
                    nudge_dir += away / norm

        norm = np.linalg.norm(nudge_dir)
        if norm < 1e-3:
            # No clear direction — try backing up in -x (away from counter)
            nudge_dir = np.array([-1.0, 0.0])
        else:
            nudge_dir /= norm

        # Apply nudge to base qpos
        qpos = get_robot_qpos(robot)
        qpos[0] += nudge_dir[0] * nudge_dist
        qpos[1] += nudge_dir[1] * nudge_dist

        print(f"    Nudging base by [{nudge_dir[0]*nudge_dist:+.3f}, "
              f"{nudge_dir[1]*nudge_dist:+.3f}]")

        # Execute the nudge
        hold = make_action(qpos[3:10], GRIPPER_OPEN, qpos[:3])
        for _ in range(40):
            step_fn(hold)

        sync_planner(planner)

    # Final check
    problems = check_start_collisions(pw, planner, target_name)
    if not problems:
        return True

    print(f"    Could not resolve start collisions after {max_attempts} attempts")
    return False
