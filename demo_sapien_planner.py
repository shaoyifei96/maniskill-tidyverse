#!/usr/bin/env python3
"""Demo: SapienPlanner (mplib 0.2) — auto-syncs with ManiSkill scene, no manual set_base_pose."""
import sys, os
import numpy as np
import torch
import sapien

import maniskill_tidyverse.tidyverse_agent
import mani_skill.envs
import gymnasium as gym
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from scipy.spatial.transform import Rotation as R

# --- Monkey-patch to handle scaled convex meshes (Robotiq gripper uses 0.001 scale) ---
import mplib.sapien_utils.conversion as _conv
from mplib.collision_detection.fcl import (
    Box, BVHModel, Capsule, CollisionObject, Convex, Cylinder, FCLObject, Halfspace, Sphere,
)
from sapien.physx import (
    PhysxArticulationLinkComponent, PhysxCollisionShapeBox, PhysxCollisionShapeCapsule,
    PhysxCollisionShapeConvexMesh, PhysxCollisionShapeCylinder, PhysxCollisionShapePlane,
    PhysxCollisionShapeSphere, PhysxCollisionShapeTriangleMesh, PhysxRigidBaseComponent,
)
from mplib import Pose as _Pose
from transforms3d.euler import euler2quat as _e2q

@staticmethod
def _patched_convert(comp):
    shapes = []
    shape_poses = []
    for shape in comp.collision_shapes:
        shape_poses.append(_Pose(shape.local_pose))
        if isinstance(shape, PhysxCollisionShapeBox):
            c_geom = Box(side=shape.half_size * 2)
        elif isinstance(shape, PhysxCollisionShapeCapsule):
            c_geom = Capsule(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= _Pose(q=_e2q(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapeConvexMesh):
            # PATCHED: handle non-unit scale by rescaling vertices
            verts = shape.vertices
            if not np.allclose(shape.scale, 1.0):
                verts = verts * np.array(shape.scale)
            c_geom = Convex(vertices=verts, faces=shape.triangles)
        elif isinstance(shape, PhysxCollisionShapeCylinder):
            c_geom = Cylinder(radius=shape.radius, lz=shape.half_length * 2)
            shape_poses[-1] *= _Pose(q=_e2q(0, np.pi / 2, 0))
        elif isinstance(shape, PhysxCollisionShapePlane):
            n = shape_poses[-1].to_transformation_matrix()[:3, 0]
            d = n.dot(shape_poses[-1].p)
            c_geom = Halfspace(n=n, d=d)
            shape_poses[-1] = _Pose()
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
    if len(shapes) == 0:
        return None
    return FCLObject(
        comp.name if isinstance(comp, PhysxArticulationLinkComponent) else _conv.convert_object_name(comp.entity),
        comp.entity.pose, shapes, shape_poses,
    )

SapienPlanningWorld.convert_physx_component = _patched_convert

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Create env ---
print("Creating PickCube env...")
env = gym.make(
    'PickCube-v1',
    render_mode='human',
    num_envs=1,
    robot_uids='tidyverse',
    control_mode='pd_joint_pos',
)
obs, info = env.reset(seed=0)
agent = env.unwrapped.agent
robot = agent.robot
links = {l.get_name(): l for l in robot.get_links()}

# --- Setup SapienPlanner ---
print("Setting up SapienPlanner (mplib 0.2 SAPIEN integration)...")

# Get the SAPIEN scene and articulation
scene = env.unwrapped.scene.sub_scenes[0]  # ManiSkill wraps scenes
physx_articulation = robot._objs[0]  # the PhysxArticulation (ManiSkill batched wrapper)

print(f"Scene type: {type(scene)}")
print(f"Articulation type: {type(physx_articulation)}")

# Create planning world from SAPIEN scene
planning_world = SapienPlanningWorld(scene, [physx_articulation])
# Link names are prefixed by SAPIEN scene — find the eef link name
eef_link_name = [n for n in planning_world.get_articulation_names()]
art_name = eef_link_name[0] if eef_link_name else None
art = planning_world.get_articulation(art_name) if art_name else None
link_names = art.get_pinocchio_model().get_link_names() if art else []
eef_name = [n for n in link_names if n.endswith('_eef')][0]
print(f"Using move_group: {eef_name}")
print(f"All links: {link_names}")
planner = SapienPlanner(planning_world, move_group=eef_name)

# Auto-sync from simulation — this is the key: no manual set_base_pose needed!
planner.update_from_simulation()
print("✅ SapienPlanner created and synced from simulation")

# Get current state — SapienPlanner uses ALL active joints (16 DOF: 3 base + 7 arm + 6 gripper)
qpos = robot.get_qpos().cpu().numpy()[0]
print(f"Full robot qpos ({len(qpos)} DOF): base={qpos[:3]}, arm={qpos[3:10]}, gripper={qpos[10:]}")

# For SapienPlanner, pass full qpos
planner_qpos = qpos.copy()

# Print move group joint info
print(f"Move group joint indices: {planner.move_group_joint_indices}")
print(f"Joint names: {planner.user_joint_names}")

tcp_pos = links['eef'].pose.p[0].cpu().numpy()
arm_base_p = links['panda_link0'].pose.p[0].cpu().numpy()
print(f"Current TCP position:   {tcp_pos}")
print(f"panda_link0 world pos:  {arm_base_p}")

# Target: 30cm forward from arm base, 15cm below
target_pos = arm_base_p + np.array([0.30, 0.0, -0.15])
print(f"Target position: {target_pos}")

# --- Define orientations ---
orientations = {
    "Top-down":              R.from_euler('xyz', [np.pi, 0, 0]),
    "Top-down rotated 45°":  R.from_euler('xyz', [np.pi, 0, np.pi/4]),
    "Top-down rotated 90°":  R.from_euler('xyz', [np.pi, 0, np.pi/2]),
    "Tilted 45° forward":    R.from_euler('xyz', [3*np.pi/4, 0, 0]),
    "Tilted 45° sideways":   R.from_euler('xyz', [np.pi, np.pi/4, 0]),
    "Horizontal forward":    R.from_euler('xyz', [np.pi/2, 0, 0]),
}

SDK_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
# Home qpos: base at current pos + SDK_HOME arm + current gripper
home_planner_qpos = np.concatenate([qpos[:3], SDK_HOME, qpos[10:]])


def render_steps(env, n=30):
    """Hold position for n steps."""
    qpos_now = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
    # pd_joint_pos action: [arm(7), gripper(1), base_vel(3)] = 11
    action = np.concatenate([qpos_now[3:10], [qpos_now[10]], [0, 0, 0]])
    for _ in range(n):
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        env.render()


def execute_trajectory(env, traj, gripper_val):
    """Execute planner trajectory. traj has move_group joints only."""
    n_mg = len(planner.move_group_joint_indices)
    print(f"  Trajectory shape: {traj.shape}, move_group size: {n_mg}")
    for i in range(traj.shape[0]):
        # traj columns correspond to move_group joints (base+arm, 10 DOF)
        # Extract arm joints from trajectory (indices 3-9 of move_group = arm 7 DOF)
        arm_joints = traj[i, 3:10]  # skip base joints in trajectory
        # pd_joint_pos action: [arm(7), gripper(1), base_vel(3)]
        action = np.concatenate([arm_joints, [gripper_val], [0, 0, 0]])
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        env.render()


render_steps(env, 60)

# --- Plan and execute each orientation ---
successes = 0
total = len(orientations)
current_planner_qpos = planner_qpos.copy()

for name, rot in orientations.items():
    quat_xyzw = rot.as_quat()
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    target_pose = sapien.Pose(p=target_pos, q=quat_wxyz)

    print(f"\n--- {name} ---")

    # Re-sync from simulation before each plan
    planner.update_from_simulation()

    # Lock base joints + gripper joints, free arm only
    # mask length = total user joints (16): base(3) + arm(7) + gripper(6)
    n_joints = len(planner.user_joint_names)
    mask = [True]*3 + [False]*7 + [True]*6  # lock base+gripper, free arm
    
    result = planner.plan_pose(
        target_pose,
        current_planner_qpos,
        mask=mask,
        time_step=env.unwrapped.control_timestep,
    )

    if result['status'] == 'Success':
        traj = result['position']
        print(f"  ✅ Plan found! {traj.shape[0]} waypoints, shape {traj.shape}")

        gripper_val = qpos[10]
        execute_trajectory(env, traj, gripper_val)

        actual_pos = links['eef'].pose.p[0].cpu().numpy()
        actual_q = links['eef'].pose.q[0].cpu().numpy()
        pos_error_mm = np.linalg.norm(actual_pos - target_pos) * 1000
        dot = abs(np.dot(actual_q, quat_wxyz))
        angle_error_deg = 2 * np.degrees(np.arccos(min(dot, 1.0)))
        print(f"  Position error: {pos_error_mm:.1f}mm")
        print(f"  Orientation error: {angle_error_deg:.1f}°")

        render_steps(env, 120)

        # Return home
        new_qpos = robot.get_qpos().cpu().numpy()[0]

        planner.update_from_simulation()
        ret = planner.plan_qpos(
            [home_planner_qpos], new_qpos,
            time_step=env.unwrapped.control_timestep,
        )
        if ret['status'] == 'Success':
            execute_trajectory(env, ret['position'], gripper_val)
            current_planner_qpos = home_planner_qpos.copy()
        else:
            print(f"  ⚠️ Return plan failed: {ret['status']}")
            current_planner_qpos = new_qpos.copy()

        render_steps(env, 60)
        successes += 1
    else:
        print(f"  ❌ Planning failed: {result['status']}")

print(f"\n{'='*40}")
print(f"Results: {successes}/{total} orientations reached")
print("Done!")
env.close()
