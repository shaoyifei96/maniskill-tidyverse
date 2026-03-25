#!/usr/bin/env python3
"""Test whole-body collision-free planning to reach fixture handles in RoboCasa kitchen.
Computes handle positions geometrically from door/drawer links.
No ACM relaxation — real obstacle avoidance."""
import sys, os, signal, time, json
import numpy as np
import torch, sapien, cv2

def timeout_handler(signum, frame): raise TimeoutError("timeout")
signal.signal(signal.SIGALRM, timeout_handler)

import maniskill_tidyverse.tidyverse_agent, mani_skill.envs, gymnasium as gym
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mplib.collision_detection.fcl import *
from sapien.physx import *
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv

# Monkey-patch for Robotiq scaled meshes
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


def compute_handle_pos(fxt, door_link):
    """Compute approximate handle position from fixture geometry.
    Handle is on the front face of the door, at the door link position.
    Front direction depends on fixture yaw.
    """
    fxt_yaw = fxt.euler[2] if len(fxt.euler) > 2 else 0.0
    door_pos = door_link.pose.p[0].cpu().numpy()

    # Front direction: in fixture's local frame, front is -Y
    # After yaw rotation: front = R(yaw) @ [0, -1, 0]
    front_dir = np.array([np.sin(fxt_yaw), -np.cos(fxt_yaw), 0.0])

    # Handle is on the front face: door_pos + small offset in front direction
    # The door link is already at the front face (hinge position)
    handle_pos = door_pos + front_dir * 0.02  # 2cm in front of door surface
    return handle_pos, front_dir, fxt_yaw


def compute_robot_pose(handle_pos, front_dir, fxt_yaw, standoff=0.60):
    """Compute robot base position to stand in front of fixture."""
    robot_xy = handle_pos[:2] + front_dir[:2] * standoff
    # Robot faces the fixture (yaw points toward fixture)
    robot_yaw = fxt_yaw + np.pi
    robot_yaw = (robot_yaw + np.pi) % (2 * np.pi) - np.pi
    return robot_xy, robot_yaw


def compute_grasp_orientation(fxt_yaw):
    """Forward grasp orientation facing into the fixture."""
    # EEF Z should point in -front_dir (into the fixture)
    # forward grasp = Ry=90° rotated by fixture yaw
    rot = R.from_euler('xyz', [0, np.pi/2, fxt_yaw])
    q = rot.as_quat()
    return np.array([q[3], q[0], q[1], q[2]])  # wxyz


# --- Create env ---
print("Creating RoboCasa env...")
env = gym.make('RoboCasaKitchen-v1', num_envs=1, robot_uids='tidyverse',
               control_mode='pd_joint_pos', render_mode='rgb_array')
obs, info = env.reset(seed=0)
robot = env.unwrapped.agent.robot
links = {l.get_name(): l for l in robot.get_links()}
scene_sapien = env.unwrapped.scene.sub_scenes[0]

# Get fixture data
sb = env.unwrapped.scene_builder
sd = sb.scene_data[0]
fixtures = sd['fixtures']

# --- Collect targets from articulated fixtures ---
targets = []
for name, fxt in fixtures.items():
    if not getattr(fxt, 'is_articulation', False):
        continue
    fxt_type = type(fxt).__name__
    # Skip stoves (knobs, not handles)
    if fxt_type == 'Stove':
        continue

    art = fxt.articulation
    if art is None:
        continue

    fxt_links = art.get_links()
    # Find door/drawer links (not dummy_root, not object)
    for link in fxt_links:
        lname = link.get_name()
        if lname in ('dummy_root_0', 'object'):
            continue
        if 'knob' in lname.lower():
            continue

        handle_pos, front_dir, fxt_yaw = compute_handle_pos(fxt, link)
        targets.append({
            'fixture': name,
            'fixture_type': fxt_type,
            'link_name': lname,
            'handle_pos': handle_pos,
            'front_dir': front_dir,
            'fixture_yaw': fxt_yaw,
        })

print(f"Found {len(targets)} interactable links across fixtures")
print(f"Types: {dict(zip(*np.unique([t['fixture_type'] for t in targets], return_counts=True)))}")

# --- Setup planner ---
pw = SapienPlanningWorld(scene_sapien, [robot._objs[0]])
art_names = pw.get_articulation_names()
robot_art_name = [n for n in art_names if 'tidyverse' in n.lower()][0]
pinocchio = pw.get_articulation(robot_art_name).get_pinocchio_model()
eef_name = [l for l in pinocchio.get_link_names() if 'eef' in l][0]
planner = SapienPlanner(pw, move_group=eef_name)

# NO ACM relaxation

# --- Test each target ---
outdir = os.path.join(os.path.dirname(__file__), 'handle_reach_images')
os.makedirs(outdir, exist_ok=True)

results = []
zero_action = torch.zeros(11, dtype=torch.float32).unsqueeze(0)

# Get robot root pose (base qpos is offset from this)
root_p = robot.pose.p[0].cpu().numpy()
root_q = robot.pose.q[0].cpu().numpy()
from scipy.spatial.transform import Rotation as R_root
root_yaw = R_root.from_quat([root_q[1], root_q[2], root_q[3], root_q[0]]).as_euler('xyz')[2]
print(f"Robot root: [{root_p[0]:.2f}, {root_p[1]:.2f}], yaw={np.degrees(root_yaw):.0f}°")

# Limit to avoid very long runs — test a diverse subset
# Pick one target per fixture (first door/drawer link)
seen_fixtures = set()
test_targets = []
for t in targets:
    if t['fixture'] not in seen_fixtures:
        seen_fixtures.add(t['fixture'])
        test_targets.append(t)

print(f"Testing {len(test_targets)} fixtures (one link each)")

for idx, tgt in enumerate(test_targets):
    fname = tgt['fixture']
    ftype = tgt['fixture_type']
    lname = tgt['link_name']
    hpos = tgt['handle_pos']
    front_dir = tgt['front_dir']
    fyaw = tgt['fixture_yaw']

    print(f"\n[{idx+1}/{len(test_targets)}] {fname} ({ftype}) — {lname}")
    print(f"  Handle pos: [{hpos[0]:.2f}, {hpos[1]:.2f}, {hpos[2]:.2f}]")
    print(f"  Wall yaw: {np.degrees(fyaw):.0f}°")

    # Compute robot pose
    robot_xy, robot_yaw = compute_robot_pose(hpos, front_dir, fyaw, standoff=0.55)
    print(f"  Robot: [{robot_xy[0]:.2f}, {robot_xy[1]:.2f}], yaw={np.degrees(robot_yaw):.0f}°")

    # Teleport robot (convert world coords to qpos offsets from root)
    qpos = robot.get_qpos().cpu().numpy()[0]
    cos_ry = np.cos(-root_yaw)
    sin_ry = np.sin(-root_yaw)
    dx = robot_xy[0] - root_p[0]
    dy = robot_xy[1] - root_p[1]
    qpos[0] = cos_ry * dx - sin_ry * dy
    qpos[1] = sin_ry * dx + cos_ry * dy
    qpos[2] = robot_yaw - root_yaw
    qpos[3:10] = SDK_HOME
    qpos[10:] = 0.0
    robot.set_qpos(torch.tensor(qpos, dtype=torch.float32).unsqueeze(0))
    # Step with SDK_HOME as action target so PD controller holds the pose
    hold_action = np.concatenate([SDK_HOME, [0.0], [0, 0, 0]])  # arm(7) + gripper(1) + base_vel(3)
    hold_action_t = torch.tensor(hold_action, dtype=torch.float32).unsqueeze(0)
    for _ in range(10):
        env.step(hold_action_t)

    # Render after teleport
    img = env.render()
    if isinstance(img, torch.Tensor): img = img.cpu().numpy()
    if img.ndim == 4: img = img[0]
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    label = f"{fname} ({ftype}) - {lname}"
    cv2.putText(img_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    cv2.putText(img_bgr, f"Wall: {np.degrees(fyaw):.0f}deg | Teleported", (10, 55),
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    safe_name = f"{idx:02d}_{fname}_{lname}".replace(' ', '_')
    teleport_path = os.path.join(outdir, f"{safe_name}_teleport.png")
    cv2.imwrite(teleport_path, img_bgr)
    print(f"  📸 Saved teleport: {os.path.basename(teleport_path)}")

    # Update planner with new robot position
    try:
        planner.update_from_simulation()
    except RuntimeError as e:
        print(f"  ⚠️ Planner sync failed: {e}")
        pw = SapienPlanningWorld(scene_sapien, [robot._objs[0]])
        planner = SapienPlanner(pw, move_group=eef_name)

    current_qpos = robot.get_qpos().cpu().numpy()[0]

    # Grasp orientation
    q_wxyz = compute_grasp_orientation(fyaw)

    # Target with small approach offset
    target_pos = hpos + front_dir * 0.03  # 3cm offset from handle
    target = MPPose(p=target_pos, q=q_wxyz)

    # Plan — whole body, NO ACM relaxation
    signal.alarm(45)
    try:
        t0 = time.time()
        result = planner.plan_pose(
            target, current_qpos,
            planning_time=15.0,
        )
        dt = time.time() - t0
        status = result['status']
        print(f"  Result: {status} ({dt:.1f}s)")

        rec = {
            'fixture': fname, 'type': ftype, 'link': lname,
            'handle_pos': hpos.tolist(), 'wall_yaw': np.degrees(fyaw),
            'status': status, 'time': round(dt, 1),
        }

        if status == 'Success':
            traj = result['position']
            print(f"  Trajectory: {traj.shape[0]} waypoints")
            rec['waypoints'] = traj.shape[0]

            # Check base motion
            base_start = current_qpos[:3]
            base_end = traj[-1, :3]
            base_delta = np.abs(base_end - base_start)
            print(f"  Base delta: dx={base_delta[0]:.3f}m dy={base_delta[1]:.3f}m dyaw={np.degrees(base_delta[2]):.1f}°")

            # Execute
            for i in range(traj.shape[0]):
                arm_q = traj[i, 3:10]
                action = np.concatenate([arm_q, [current_qpos[10]], [0, 0, 0]])
                obs, _, _, _, _ = env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

            for _ in range(20):
                obs, _, _, _, _ = env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))

            # Render
            img = env.render()
            if isinstance(img, torch.Tensor): img = img.cpu().numpy()
            if img.ndim == 4: img = img[0]
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            label = f"{fname} ({ftype})"
            cv2.putText(img_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(img_bgr, f"Link: {lname} | Wall: {np.degrees(fyaw):.0f}deg",
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
            cv2.putText(img_bgr, f"Plan: {dt:.1f}s | wpts: {traj.shape[0]}",
                       (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

            safe_name = f"{idx:02d}_{fname}_{lname}".replace(' ', '_')
            filepath = os.path.join(outdir, f"{safe_name}.png")
            cv2.imwrite(filepath, img_bgr)
            print(f"  ✅ Saved: {os.path.basename(filepath)}")
            rec['image'] = os.path.basename(filepath)
        else:
            print(f"  ❌ {status}")

        results.append(rec)

    except TimeoutError:
        print(f"  ❌ TIMEOUT")
        results.append({'fixture': fname, 'type': ftype, 'link': lname, 'status': 'TIMEOUT'})
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        results.append({'fixture': fname, 'type': ftype, 'link': lname, 'status': f'ERROR: {e}'})
    signal.alarm(0)

# --- Summary ---
env.close()
successes = [r for r in results if r.get('status') == 'Success']
failures = [r for r in results if r.get('status') != 'Success']
print(f"\n{'='*60}")
print(f"RESULTS: {len(successes)}/{len(results)} fixtures reached")
print(f"{'='*60}")

by_type = {}
for r in results:
    t = r.get('type', '?')
    by_type.setdefault(t, {'ok': 0, 'fail': 0})
    if r['status'] == 'Success': by_type[t]['ok'] += 1
    else: by_type[t]['fail'] += 1

print(f"\nBy type:")
for t, counts in sorted(by_type.items()):
    total = counts['ok'] + counts['fail']
    print(f"  {t}: {counts['ok']}/{total}")

print(f"\n✅ Successes ({len(successes)}):")
for r in successes:
    print(f"  {r['fixture']} ({r['type']}) {r['link']} — {r['time']}s")
print(f"\n❌ Failures ({len(failures)}):")
for r in failures:
    print(f"  {r['fixture']} ({r['type']}) {r['link']} — {r['status']}")

with open(os.path.join(outdir, 'results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to {outdir}/results.json")
