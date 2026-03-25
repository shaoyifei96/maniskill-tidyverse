#!/usr/bin/env python3
"""Test: Spawn objects on RoboCasa counter, plan to reach them with SapienPlanner."""
import sys, os, signal, time
import numpy as np
import torch, sapien

def timeout_handler(signum, frame): raise TimeoutError("timeout")
signal.signal(signal.SIGALRM, timeout_handler)

import maniskill_tidyverse.tidyverse_agent, mani_skill.envs, gymnasium as gym
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

# --- Find nearest counter ---
counters = [(np.linalg.norm(np.array(f.pos[:2]) - robot_pos[:2]), n, f) 
            for n, f in fixtures.items() if isinstance(f, Counter) and f.size[0] > 0.3]
counters.sort()
counter_name, counter = counters[0][1], counters[0][2]
surface_z = counter.pos[2] + counter.size[2] / 2
print(f"Counter: {counter_name}, surface_z={surface_z:.3f}")
print(f"Arm base: {arm_base}, TCP: {tcp}")

# --- Place objects within arm reach ---
# Place objects near the edge of the counter closest to the robot
# Counter is oriented along x, robot is roughly in -y direction from counter
# We want objects near the robot side of the counter
obj_y = counter.pos[1] - counter.size[1] / 2 + 0.10  # near edge closest to robot
# And centered around the arm's x position  
obj_x_center = arm_base[0]

objects_info = []
for i, (name, color, half_size) in enumerate([
    ("red_cube",    [1,0,0,1], [0.02, 0.02, 0.02]),
    ("green_box",   [0,1,0,1], [0.025, 0.015, 0.03]),
    ("blue_cube",   [0,0,1,1], [0.02, 0.02, 0.02]),
]):
    obj_x = obj_x_center + (i - 1) * 0.12
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
    print(f"  {name}: pos={obj_pos}, dist_from_arm={dist_arm:.2f}m")

# --- Setup SapienPlanner ---
print("\nSetting up SapienPlanner...")
signal.alarm(30)
pw = SapienPlanningWorld(scene, [robot._objs[0]])
eef_name = [l for l in pw.get_planned_articulations()[0].get_pinocchio_model().get_link_names() if 'eef' in l][0]
planner = SapienPlanner(pw, move_group=eef_name)
signal.alarm(0)

# Relax ACM for kitchen fixtures (allow robot to "pass through" static fixtures during IK sampling)
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

# Also allow collision with static scene objects (walls, floors, counters)
for on in pw.get_object_names():
    # Don't allow collision with our placed objects!
    if on in ['red_cube', 'green_box', 'blue_cube']:
        continue
    for rl in robot_link_names:
        acm.set_entry(rl, on, True)

print(f"ACM relaxed for {len(art_names)-1} articulations + static objects")
print(f"Collision-checked objects: {[n for n in pw.get_object_names() if n in ['red_cube','green_box','blue_cube']]}")

# --- Plan to each object ---
qpos = robot.get_qpos().cpu().numpy()[0]
rot = R.from_euler('xyz', [np.pi, 0, 0])
q = rot.as_quat(); q_wxyz = [q[3], q[0], q[1], q[2]]
mask = np.array([True]*3 + [False]*7 + [True]*6)  # arm-only

print("\n=== Planning to reach objects ===")
successes = 0
for name, obj_pos, dist_arm in objects_info:
    # Target: 8cm above object, top-down orientation
    target_pos = obj_pos + [0, 0, 0.08]
    target = MPPose(p=target_pos, q=q_wxyz)
    
    print(f"\n--- {name} at {obj_pos} (dist={dist_arm:.2f}m) ---")
    print(f"  Target: {target_pos}")
    
    signal.alarm(15)
    t0 = time.time()
    try:
        # Arm-only
        result = planner.plan_pose(target, qpos, mask=mask, planning_time=5.0)
        dt = time.time() - t0
        
        if result['status'] == 'Success':
            traj = result['position']
            print(f"  ✅ ARM-ONLY: {traj.shape[0]} waypts ({dt:.2f}s)")
            
            # Execute
            for i in range(traj.shape[0]):
                arm_q = traj[i, 3:10]
                action = np.concatenate([arm_q, [qpos[10]], [0,0,0]])
                env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
            
            final_tcp = links['eef'].pose.p[0].cpu().numpy()
            err_mm = np.linalg.norm(final_tcp - target_pos) * 1000
            print(f"  Position error: {err_mm:.1f}mm")
            qpos = robot.get_qpos().cpu().numpy()[0]  # update for next plan
            successes += 1
        else:
            print(f"  ❌ ARM-ONLY failed: {result['status']} ({dt:.2f}s)")
            
            # Try whole-body
            t1 = time.time()
            result2 = planner.plan_pose(target, qpos, planning_time=5.0)  # no mask
            dt2 = time.time() - t1
            if result2['status'] == 'Success':
                traj2 = result2['position']
                base_delta = np.linalg.norm(traj2[-1,:3] - traj2[0,:3])
                print(f"  ✅ WHOLE-BODY: {traj2.shape[0]} waypts, base_delta={base_delta:.3f}m ({dt2:.2f}s)")
                successes += 1
            else:
                print(f"  ❌ WHOLE-BODY also failed: {result2['status']} ({dt2:.2f}s)")
    except TimeoutError:
        print(f"  ❌ TIMEOUT")
    signal.alarm(0)

print(f"\n{'='*50}")
print(f"Results: {successes}/{len(objects_info)} objects reachable")
env.close()
print("Done!")
