#!/usr/bin/env python3
"""Demo: plan to same position with different EE orientations using arm-only planner (mplib 0.2)."""
import sys, os
import numpy as np
import torch
import sapien

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent
import mani_skill.envs
import gymnasium as gym
import mplib
from scipy.spatial.transform import Rotation as R

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

# --- Setup arm-only planner (mplib 0.2 API) ---
print("Setting up arm-only planner (mplib 0.2)...")
planner = mplib.Planner(
    urdf=os.path.join(SCRIPT_DIR, 'tidyverse_arm_planning.urdf'),
    srdf=os.path.join(SCRIPT_DIR, 'tidyverse_arm_planning_mplib.srdf'),
    move_group='eef',
)

# Sync planner base pose to base_link world position
links = {l.get_name(): l for l in robot.get_links()}
base_p = links['base_link'].pose.p[0].cpu().numpy()
base_q = links['base_link'].pose.q[0].cpu().numpy()
from mplib import Pose as MPPose
planner.robot.set_base_pose(MPPose(p=base_p, q=base_q))

# Get current state
qpos = robot.get_qpos().cpu().numpy()[0]
arm_qpos = qpos[3:10]
gripper_qpos = qpos[10:]
planner_qpos = np.concatenate([arm_qpos, gripper_qpos])

# Get current TCP position
eef_pose = links['eef'].pose
tcp_pos = eef_pose.p[0].cpu().numpy()
print(f"Current TCP position: {tcp_pos}")

# Target: 35cm forward, ~table height relative to base_link
target_pos = np.array([base_p[0] + 0.35, base_p[1], base_p[2] + 0.45])
print(f"Target position: {target_pos}")

# --- Define orientations to test ---
orientations = {
    "Top-down":              R.from_euler('xyz', [np.pi, 0, 0]),
    "Top-down rotated 45°":  R.from_euler('xyz', [np.pi, 0, np.pi/4]),
    "Top-down rotated 90°":  R.from_euler('xyz', [np.pi, 0, np.pi/2]),
    "Tilted 45° forward":    R.from_euler('xyz', [3*np.pi/4, 0, 0]),
    "Tilted 45° sideways":   R.from_euler('xyz', [np.pi, np.pi/4, 0]),
    "Horizontal forward":    R.from_euler('xyz', [np.pi/2, 0, 0]),
}

SDK_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
home_planner_qpos = np.concatenate([SDK_HOME, gripper_qpos])


def render_steps(env, n=30):
    """Hold position so viewer updates."""
    qpos_now = env.unwrapped.agent.robot.get_qpos().cpu().numpy()[0]
    action = np.concatenate([qpos_now[3:10], [qpos_now[10]], [0, 0, 0]])
    for _ in range(n):
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        env.render()


def execute_trajectory(env, traj, gripper_val):
    """Execute arm-only trajectory. traj shape: (N, 7) arm joints."""
    for i in range(traj.shape[0]):
        action = np.concatenate([traj[i, :7], [gripper_val], [0, 0, 0]])
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        env.render()


# Let viewer initialize
render_steps(env, 60)

# --- Plan and execute each orientation ---
successes = 0
total = len(orientations)
current_planner_qpos = planner_qpos.copy()

for name, rot in orientations.items():
    quat_xyzw = rot.as_quat()  # scipy: [x, y, z, w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

    target_pose = sapien.Pose(p=target_pos, q=quat_wxyz)

    print(f"\n--- {name} ---")
    print(f"  Target: pos={target_pos}, quat(wxyz)={quat_wxyz}")

    result = planner.plan_pose(
        target_pose,
        current_planner_qpos,
        time_step=env.unwrapped.control_timestep,
    )

    if result['status'] == 'Success':
        traj = result['position']
        print(f"  ✅ Plan found! {traj.shape[0]} waypoints, shape {traj.shape}")

        gripper_val = qpos[10]
        execute_trajectory(env, traj, gripper_val)

        # Check actual TCP
        actual_pos = links['eef'].pose.p[0].cpu().numpy()
        actual_q = links['eef'].pose.q[0].cpu().numpy()
        pos_error_mm = np.linalg.norm(actual_pos - target_pos) * 1000

        # Orientation error (angle between quaternions)
        dot = abs(np.dot(actual_q, quat_wxyz))
        angle_error_deg = 2 * np.degrees(np.arccos(min(dot, 1.0)))
        print(f"  Position error: {pos_error_mm:.1f}mm")
        print(f"  Orientation error: {angle_error_deg:.1f}°")

        # Pause so user can see the pose
        render_steps(env, 120)

        # Return to home config
        new_arm_qpos = robot.get_qpos().cpu().numpy()[0][3:10]
        new_planner_qpos = np.concatenate([new_arm_qpos, gripper_qpos])

        ret = planner.plan_qpos(
            [home_planner_qpos], new_planner_qpos,
            time_step=env.unwrapped.control_timestep,
        )
        if ret['status'] == 'Success':
            execute_trajectory(env, ret['position'], gripper_val)
            current_planner_qpos = home_planner_qpos.copy()
        else:
            print(f"  ⚠️ Return plan failed, staying here")
            current_planner_qpos = new_planner_qpos.copy()

        render_steps(env, 60)
        successes += 1
    else:
        print(f"  ❌ Planning failed: {result['status']}")

print(f"\n{'='*40}")
print(f"Results: {successes}/{total} orientations reached")
print("Done! Closing env.")
env.close()
