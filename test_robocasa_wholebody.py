#!/usr/bin/env python3
"""Test: Whole-body planning in RoboCasa — base + arm to reach distant objects."""
import sys, os, signal, time
import numpy as np
import torch, sapien
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def timeout_handler(signum, frame): raise TimeoutError("timeout")
signal.signal(signal.SIGALRM, timeout_handler)

import tidyverse_agent, mani_skill.envs, gymnasium as gym
from mani_skill.utils.scene_builder.robocasa.fixtures.counter import Counter
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mplib.collision_detection.fcl import *
from sapien.physx import *
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R
import mplib.sapien_utils.conversion as _conv

# Monkey-patch for scaled convex meshes
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

# --- Create env ---
print("Creating RoboCasa env...")
env = gym.make('RoboCasaKitchen-v1', num_envs=1, robot_uids='tidyverse', control_mode='pd_joint_pos')
obs, info = env.reset(seed=0)

robot = env.unwrapped.agent.robot
links = {l.get_name(): l for l in robot.get_links()}
robot_pos = robot.pose.p[0].cpu().numpy()
tcp = links['eef'].pose.p[0].cpu().numpy()
arm_base = links['panda_link0'].pose.p[0].cpu().numpy()

sb = env.unwrapped.scene_builder
fixtures = sb.scene_data[0]['fixtures']
scene = env.unwrapped.scene.sub_scenes[0]

print(f"Robot root: {robot_pos}")
print(f"Arm base: {arm_base}")
print(f"TCP: {tcp}")

# --- Place objects at varying distances: near, medium, far ---
counters = [(np.linalg.norm(np.array(f.pos[:2]) - robot_pos[:2]), n, f) 
            for n, f in fixtures.items() if isinstance(f, Counter) and f.size[0] > 0.3]
counters.sort()

# Use nearest counter but place objects at different distances
counter = counters[0][2]
surface_z = counter.pos[2] + counter.size[2] / 2
print(f"\nCounter: {counters[0][1]}, surface_z={surface_z:.3f}")

# Place objects from near (arm-only reachable) to far (needs base movement)
objects_info = []
obj_y = counter.pos[1] - counter.size[1] / 2 + 0.10

test_objects = [
    ("near_cube",   [1,0,0,1], [0.02]*3, arm_base[0] + 0.0),      # ~0.5m from arm
    ("mid_cube",    [0,1,0,1], [0.02]*3, arm_base[0] + 0.5),      # ~0.7m from arm  
    ("far_cube",    [0,0,1,1], [0.02]*3, arm_base[0] + 1.0),      # ~1.1m from arm (needs base)
    ("vfar_cube",   [1,1,0,1], [0.02]*3, arm_base[0] + 1.5),      # ~1.6m from arm (definitely needs base)
]

for name, color, half_size, obj_x in test_objects:
    obj_z = surface_z + half_size[2] + 0.002
    builder = scene.create_actor_builder()
    builder.add_box_collision(half_size=np.array(half_size))
    builder.add_box_visual(half_size=np.array(half_size),
                          material=sapien.render.RenderMaterial(base_color=color))
    actor = builder.build(name=name)
    actor.set_pose(sapien.Pose(p=[obj_x, obj_y, obj_z]))
    obj_pos = np.array([obj_x, obj_y, obj_z])
    dist_arm = np.linalg.norm(arm_base - obj_pos)
    objects_info.append((name, obj_pos, dist_arm))
    print(f"  {name}: pos=({obj_x:.2f}, {obj_y:.2f}, {obj_z:.3f}), dist_from_arm={dist_arm:.2f}m")

# --- Setup SapienPlanner ---
print("\nSetting up SapienPlanner...")
signal.alarm(30)
pw = SapienPlanningWorld(scene, [robot._objs[0]])
eef_name = [l for l in pw.get_planned_articulations()[0].get_pinocchio_model().get_link_names() if 'eef' in l][0]
planner = SapienPlanner(pw, move_group=eef_name)
signal.alarm(0)

# Relax ACM for kitchen fixtures
acm = pw.get_allowed_collision_matrix()
art_names = pw.get_articulation_names()
pm = planner.pinocchio_model
robot_link_names = pm.get_link_names()
planned_name = [n for n in art_names if 'tidyverse' in n][0]

for an in art_names:
    if an != planned_name:
        ul = pw.get_articulation(an).get_pinocchio_model().get_link_names()
        for rl in robot_link_names:
            for u in ul:
                acm.set_entry(rl, u, True)

for on in pw.get_object_names():
    # Keep collision checks for our objects
    if any(on.startswith(prefix) for prefix in ['near_', 'mid_', 'far_', 'vfar_']):
        continue
    for rl in robot_link_names:
        acm.set_entry(rl, on, True)

print(f"ACM relaxed")

# --- Test planning ---
qpos = robot.get_qpos().cpu().numpy()[0]
rot = R.from_euler('xyz', [np.pi, 0, 0])
q = rot.as_quat(); q_wxyz = [q[3], q[0], q[1], q[2]]
mask_arm = np.array([True]*3 + [False]*7 + [True]*6)

# Check base joint limits
jlimits = np.concatenate(pm.get_joint_limits())
jnames = pm.get_joint_names()
print(f"\nBase joint limits:")
for i in range(3):
    print(f"  {jnames[i]}: [{jlimits[i,0]:.2f}, {jlimits[i,1]:.2f}]")
print("Note: Can't tighten via API — limits come from auto-generated URDF")

print("\n" + "="*60)
print("PLANNING TESTS: ARM-ONLY vs WHOLE-BODY")
print("="*60)

for name, obj_pos, dist_arm in objects_info:
    target_pos = obj_pos + [0, 0, 0.08]
    target = MPPose(p=target_pos, q=q_wxyz)
    reachable = dist_arm < 0.80
    
    print(f"\n--- {name} | dist={dist_arm:.2f}m | arm-reachable={'yes' if reachable else 'NO'} ---")
    
    # Test 1: Arm-only
    signal.alarm(15)
    t0 = time.time()
    try:
        result_arm = planner.plan_pose(target, qpos, mask=mask_arm, planning_time=5.0)
        dt = time.time() - t0
        if result_arm['status'] == 'Success':
            traj = result_arm['position']
            print(f"  ARM-ONLY: ✅ {traj.shape[0]} waypts ({dt:.2f}s)")
        else:
            print(f"  ARM-ONLY: ❌ {result_arm['status']} ({dt:.2f}s)")
    except TimeoutError:
        print(f"  ARM-ONLY: ❌ TIMEOUT")
    signal.alarm(0)
    
    # Test 2: Whole-body (no mask)
    signal.alarm(15)
    t1 = time.time()
    try:
        result_wb = planner.plan_pose(target, qpos, planning_time=5.0)
        dt1 = time.time() - t1
        if result_wb['status'] == 'Success':
            traj_wb = result_wb['position']
            base_start = traj_wb[0, :3]
            base_end = traj_wb[-1, :3]
            base_delta = np.linalg.norm(base_end[:2] - base_start[:2])
            yaw_delta = abs(base_end[2] - base_start[2])
            print(f"  WHOLE-BODY: ✅ {traj_wb.shape[0]} waypts ({dt1:.2f}s)")
            print(f"    Base moved: xy={base_delta:.3f}m, yaw={np.degrees(yaw_delta):.1f}°")
            print(f"    Base: ({base_start[0]:.3f},{base_start[1]:.3f}) → ({base_end[0]:.3f},{base_end[1]:.3f})")
        else:
            print(f"  WHOLE-BODY: ❌ {result_wb['status']} ({dt1:.2f}s)")
    except TimeoutError:
        print(f"  WHOLE-BODY: ❌ TIMEOUT")
    signal.alarm(0)

    # Test 3: Execute whole-body if successful (on the last object)
    if name == "far_cube" and result_wb.get('status') == 'Success':
        print(f"\n  >>> Executing whole-body trajectory for {name}...")
        traj_exec = result_wb['position']
        for i in range(traj_exec.shape[0]):
            # Whole-body: need to set base position directly (can't use base velocity control for planned positions)
            # In pd_joint_pos mode: action = [arm(7), gripper(1), base_vel(3)]
            # For base, we need velocity commands that move base to planned positions
            # Simpler: just set qpos directly for validation
            pass
        
        # Just verify by setting final qpos
        final_qpos = np.zeros_like(qpos)
        final_qpos[:3] = traj_exec[-1, :3]  # base from trajectory
        final_qpos[3:10] = traj_exec[-1, 3:10]  # arm from trajectory
        final_qpos[10:] = qpos[10:]  # keep gripper
        
        # Can't easily execute base motion in pd_joint_pos mode
        # But we can verify the FK
        planner.robot.set_qpos(final_qpos, full=True)
        pm.compute_forward_kinematics(final_qpos)
        eef_idx = pm.get_link_names().index(eef_name)
        fk_pose = pm.get_link_pose(eef_idx)
        bp = planner.robot.get_base_pose()
        world_eef = bp * fk_pose  # transform to world frame
        err = np.linalg.norm(world_eef.p - target_pos) * 1000
        print(f"  FK verification: target={target_pos}, FK={world_eef.p}, error={err:.1f}mm")

env.close()
print("\nDone!")
