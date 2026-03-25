#!/usr/bin/env python3
"""Visualize EEF orientations in PickCube — render each pose and save images."""
import sys, os, signal, time
import numpy as np
import torch, sapien

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

# Monkey-patch
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

# --- Create env with rendering ---
print("Creating PickCube env...")
env = gym.make(
    'PickCube-v1',
    num_envs=1,
    robot_uids='tidyverse',
    control_mode='pd_joint_pos',
    render_mode='rgb_array',
    sensor_configs=dict(shader_pack="default"),
)
obs, info = env.reset(seed=0)
robot = env.unwrapped.agent.robot
links = {l.get_name(): l for l in robot.get_links()}

# Setup planner
scene = env.unwrapped.scene.sub_scenes[0]
pw = SapienPlanningWorld(scene, [robot._objs[0]])
eef_name = [l for l in pw.get_planned_articulations()[0].get_pinocchio_model().get_link_names() if 'eef' in l][0]
planner = SapienPlanner(pw, move_group=eef_name)

qpos = robot.get_qpos().cpu().numpy()[0]
arm_base = links['panda_link0'].pose.p[0].cpu().numpy()
mask = np.array([True]*3 + [False]*7 + [True]*6)

# Target position: 30cm forward, 15cm below arm base
target_pos = arm_base + np.array([0.35, 0.0, -0.05])
print(f"Arm base: {arm_base}")
print(f"Target pos: {target_pos}")

# --- Define orientations using different quaternion inputs ---
# SapienPlanner uses MPPose which takes quaternion in wxyz format
# We'll show Euler angles → quaternion conversion for clarity

orientations = {
    # Name: (euler_angles_xyz_rad, description)
    "1_top_down":           ([np.pi, 0, 0],       "Rx=180° (gripper pointing straight down)"),
    "2_top_down_yaw45":     ([np.pi, 0, np.pi/4], "Rx=180° Rz=45° (down, rotated 45° about Z)"),
    "3_top_down_yaw90":     ([np.pi, 0, np.pi/2], "Rx=180° Rz=90° (down, rotated 90° about Z)"),
    "4_tilt_fwd_45":        ([3*np.pi/4, 0, 0],   "Rx=135° (tilted 45° forward from down)"),
    "5_tilt_side_45":       ([np.pi, np.pi/4, 0],  "Rx=180° Ry=45° (tilted 45° sideways)"),
    "6_horizontal_fwd":     ([np.pi/2, 0, 0],     "Rx=90° (horizontal, gripper pointing forward)"),
    "7_horizontal_side":    ([np.pi/2, 0, np.pi/2], "Rx=90° Rz=90° (horizontal, pointing left)"),
    "8_identity":           ([0, 0, 0],            "Identity (gripper pointing UP)"),
}

import cv2
output_dir = os.path.join(os.path.dirname(__file__), 'orientation_images')
os.makedirs(output_dir, exist_ok=True)

SDK_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
home_qpos = qpos.copy()
home_qpos[3:10] = SDK_HOME

current_qpos = qpos.copy()
successes = []

for name, (euler_xyz, desc) in orientations.items():
    rot = R.from_euler('xyz', euler_xyz)
    q_xyzw = rot.as_quat()
    q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    
    target = MPPose(p=target_pos, q=q_wxyz)
    
    print(f"\n--- {name}: {desc} ---")
    print(f"  Euler xyz: [{np.degrees(euler_xyz[0]):.0f}°, {np.degrees(euler_xyz[1]):.0f}°, {np.degrees(euler_xyz[2]):.0f}°]")
    print(f"  Quat wxyz: [{q_wxyz[0]:.4f}, {q_wxyz[1]:.4f}, {q_wxyz[2]:.4f}, {q_wxyz[3]:.4f}]")
    
    signal.alarm(10)
    try:
        result = planner.plan_pose(target, current_qpos, mask=mask, planning_time=5.0)
        
        if result['status'] == 'Success':
            traj = result['position']
            print(f"  ✅ Plan found: {traj.shape[0]} waypoints")
            
            # Execute trajectory
            for i in range(traj.shape[0]):
                arm_q = traj[i, 3:10]
                action = np.concatenate([arm_q, [current_qpos[10]], [0, 0, 0]])
                obs, _, _, _, _ = env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
            
            # Hold and render
            arm_q = traj[-1, 3:10]
            action = np.concatenate([arm_q, [current_qpos[10]], [0, 0, 0]])
            for _ in range(30):
                obs, _, _, _, _ = env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
            
            # Capture image
            img = env.render()
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.ndim == 4:
                img = img[0]
            
            # Add text overlay
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(img_bgr, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(img_bgr, desc, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            euler_str = f"Euler: [{np.degrees(euler_xyz[0]):.0f}, {np.degrees(euler_xyz[1]):.0f}, {np.degrees(euler_xyz[2]):.0f}] deg"
            cv2.putText(img_bgr, euler_str, (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            quat_str = f"Quat wxyz: [{q_wxyz[0]:.3f}, {q_wxyz[1]:.3f}, {q_wxyz[2]:.3f}, {q_wxyz[3]:.3f}]"
            cv2.putText(img_bgr, quat_str, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            
            filepath = os.path.join(output_dir, f"{name}.png")
            cv2.imwrite(filepath, img_bgr)
            print(f"  Saved: {filepath}")
            successes.append(name)
            
            # Update qpos
            current_qpos = robot.get_qpos().cpu().numpy()[0]
            
            # Return home
            ret = planner.plan_qpos([home_qpos], current_qpos, time_step=env.unwrapped.control_timestep)
            if ret['status'] == 'Success':
                for i in range(ret['position'].shape[0]):
                    arm_q = ret['position'][i, 3:10]
                    action = np.concatenate([arm_q, [current_qpos[10]], [0, 0, 0]])
                    env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
                current_qpos = robot.get_qpos().cpu().numpy()[0]
        else:
            print(f"  ❌ {result['status']}")
    except TimeoutError:
        print(f"  ❌ TIMEOUT")
    signal.alarm(0)

# Make a grid image
print(f"\nCreating grid ({len(successes)} images)...")
imgs = []
for name in successes:
    img = cv2.imread(os.path.join(output_dir, f"{name}.png"))
    if img is not None:
        imgs.append(img)

if imgs:
    # Resize all to same size
    h, w = imgs[0].shape[:2]
    cols = min(4, len(imgs))
    rows = (len(imgs) + cols - 1) // cols
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for idx, img in enumerate(imgs):
        r, c = idx // cols, idx % cols
        grid[r*h:(r+1)*h, c*w:(c+1)*w] = img
    
    grid_path = os.path.join(output_dir, "grid_all_orientations.png")
    cv2.imwrite(grid_path, grid)
    print(f"Grid saved: {grid_path}")

env.close()
print(f"\n{len(successes)}/{len(orientations)} orientations reached. Images in {output_dir}/")
