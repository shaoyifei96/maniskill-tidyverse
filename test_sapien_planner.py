#!/usr/bin/env python3
"""Minimal test: SapienPlanner with TidyVerse in PickCube tabletop scene."""
import sys, os, signal, time
import numpy as np
import torch


# Timeout handler
def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out!")
signal.signal(signal.SIGALRM, timeout_handler)

import maniskill_tidyverse.tidyverse_agent
import mani_skill.envs
import gymnasium as gym

# --- Monkey-patch for scaled convex meshes ---
import mplib.sapien_utils.conversion as _conv
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mplib.collision_detection.fcl import (
    Box, BVHModel, Capsule, CollisionObject, Convex, Cylinder, FCLObject, Halfspace, Sphere,
)
from sapien.physx import (
    PhysxArticulationLinkComponent, PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh, PhysxCollisionShapeCylinder, PhysxCollisionShapePlane,
    PhysxCollisionShapeSphere, PhysxCollisionShapeTriangleMesh,
)
from transforms3d.euler import euler2quat
from scipy.spatial.transform import Rotation as R

@staticmethod
def _patched_convert(comp):
    shapes, shape_poses = [], []
    for shape in comp.collision_shapes:
        shape_poses.append(MPPose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox):
            c_geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            c_geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0):
                verts = verts * np.array(shape.scale)
            c_geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            c_geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= MPPose(q=euler2quat(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]
            d = n.dot(shape_poses[-1].p)
            c_geom = Halfspace(n=n, d=d)
            shape_poses[-1] = MPPose()
        elif isinstance(shape, PhysxCollisionShapeSphere):
            c_geom = Sphere(radius=shape.radius)
        elif isinstance(shape, PhysxCollisionShapeTriangleMesh):
            c_geom = BVHModel()
            c_geom.begin_model()
            c_geom.add_sub_model(vertices=shape.vertices, faces=shape.triangles)
            c_geom.end_model()
        else:
            raise TypeError(f"Unknown shape type: {type(shape)}")
        shapes.append(CollisionObject(c_geom))
    if not shapes:
        return None
    return FCLObject(
        comp.name if isinstance(comp, PhysxArticulationLinkComponent)
        else _conv.convert_object_name(comp.entity),
        comp.entity.pose, shapes, shape_poses,
    )

SapienPlanningWorld.convert_physx_component = _patched_convert

# --- Create env (headless) ---
print("Creating PickCube env (headless)...")
env = gym.make(
    'PickCube-v1',
    num_envs=1,
    robot_uids='tidyverse',
    control_mode='pd_joint_pos',
)
obs, info = env.reset(seed=0)
robot = env.unwrapped.agent.robot
links = {l.get_name(): l for l in robot.get_links()}

# --- Setup SapienPlanner ---
print("Setting up SapienPlanner...")
scene = env.unwrapped.scene.sub_scenes[0]
physx_art = robot._objs[0]

signal.alarm(30)  # 30s timeout for planner creation
try:
    planning_world = SapienPlanningWorld(scene, [physx_art])
    
    # Find eef link name (prefixed by scene)
    art_name = planning_world.get_articulation_names()[0]
    art = planning_world.get_articulation(art_name)
    link_names = art.get_pinocchio_model().get_link_names()
    joint_names = art.get_pinocchio_model().get_joint_names()
    eef_name = [n for n in link_names if 'eef' in n][0]
    
    print(f"Move group: {eef_name}")
    print(f"Joints ({len(joint_names)}): {joint_names}")
    
    planner = SapienPlanner(planning_world, move_group=eef_name)
    planner.update_from_simulation()
    print(f"✅ Planner created! Move group indices: {planner.move_group_joint_indices}")
    
    # Check for initial collisions
    collisions = planner.check_for_self_collision()
    print(f"Self-collisions: {len(collisions)}")
    env_collisions = planner.check_for_env_collision()
    print(f"Env collisions: {len(env_collisions)}")
    for c in env_collisions:
        print(f"  {c.link_name1} vs {c.object_name2}/{c.link_name2}")
    
except TimeoutError:
    print("❌ TIMEOUT during planner creation (likely DAE mesh hang)")
    env.close()
    sys.exit(1)
signal.alarm(0)

# --- Get state ---
qpos = robot.get_qpos().cpu().numpy()[0]
print(f"\nFull qpos ({len(qpos)} DOF)")
print(f"  Base:    {qpos[:3]}")
print(f"  Arm:     {qpos[3:10]}")
print(f"  Gripper: {qpos[10:]}")

tcp_pos = links[eef_name.replace('scene-0-tidyverse_', '')].pose.p[0].cpu().numpy()
arm_base = links['panda_link0'].pose.p[0].cpu().numpy()
print(f"  TCP pos: {tcp_pos}")
print(f"  Arm base: {arm_base}")

# --- Scene objects ---
obj_names = planning_world.get_object_names()
print(f"\nScene objects ({len(obj_names)}): {obj_names}")

# --- Test: plan to cube position ---
cube = env.unwrapped.cube
cube_pos = cube.pose.p[0].cpu().numpy()
print(f"\nCube position: {cube_pos}")

# Target: 5cm above cube, top-down orientation
target_pos = cube_pos + np.array([0, 0, 0.05])
rot = R.from_euler('xyz', [np.pi, 0, 0])
q_xyzw = rot.as_quat()
q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

import sapien
target_pose = sapien.Pose(p=target_pos, q=q_wxyz)
print(f"Target: pos={target_pos}, quat_wxyz={q_wxyz}")

# Mask: lock base(3) + gripper(6), free arm(7)
mask = np.array([True]*3 + [False]*7 + [True]*6)
print(f"Mask: {mask} (len={len(mask)})")

# Plan
print("\nPlanning (arm-only, 10s timeout)...")
signal.alarm(10)
try:
    t0 = time.time()
    result = planner.plan_pose(
        target_pose,
        qpos,
        mask=mask,
        time_step=env.unwrapped.control_timestep,
        planning_time=5.0,
    )
    dt = time.time() - t0
    print(f"Status: {result['status']} ({dt:.2f}s)")
    
    if result['status'] == 'Success':
        traj = result['position']
        print(f"Trajectory: {traj.shape} waypoints")
        print(f"  Start arm qpos: {traj[0, 3:10]}")
        print(f"  End arm qpos:   {traj[-1, 3:10]}")
        print(f"  Base moved: {np.any(np.abs(traj[:, :3] - traj[0, :3]) > 0.001)}")
        
        # Execute trajectory
        print("\nExecuting trajectory...")
        for i in range(traj.shape[0]):
            arm_q = traj[i, 3:10]
            gripper_val = qpos[10]
            action = np.concatenate([arm_q, [gripper_val], [0, 0, 0]])
            env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        
        # Check result
        final_tcp = links[eef_name.replace('scene-0-tidyverse_', '')].pose.p[0].cpu().numpy()
        pos_err_mm = np.linalg.norm(final_tcp - target_pos) * 1000
        print(f"✅ Position error: {pos_err_mm:.1f}mm")
    else:
        print(f"❌ Planning failed")

except TimeoutError:
    print("❌ TIMEOUT during planning (IK or RRT hung)")
signal.alarm(0)

# --- Test 2: plan_screw (straight-line motion) ---
print("\n--- Test 2: plan_screw ---")
planner.update_from_simulation()
new_qpos = robot.get_qpos().cpu().numpy()[0]

target2_pos = arm_base + np.array([0.30, 0.0, 0.10])
target2_pose = sapien.Pose(p=target2_pos, q=q_wxyz)
print(f"Target2: {target2_pos}")

signal.alarm(10)
try:
    t0 = time.time()
    result2 = planner.plan_screw(
        target2_pose,
        new_qpos,
        time_step=env.unwrapped.control_timestep,
    )
    dt = time.time() - t0
    print(f"Status: {result2['status']} ({dt:.2f}s)")
    if result2['status'] == 'Success':
        print(f"Trajectory: {result2['position'].shape}")
except TimeoutError:
    print("❌ TIMEOUT during plan_screw")
signal.alarm(0)

print("\n✅ All tests complete!")
env.close()
