#!/usr/bin/env python3
"""Quick PickCube demo with TidyVerse robot + GUI viewer."""
import sys, os, time

import maniskill_tidyverse.tidyverse_agent  # registers 'tidyverse'
import mani_skill.envs
import gymnasium as gym
import numpy as np
import torch

print("Creating PickCube env with GUI...")
env = gym.make(
    'PickCube-v1',
    render_mode='human',
    num_envs=1,
    robot_uids='tidyverse',
    control_mode='pd_ee_delta_pose',
)

obs, info = env.reset(seed=0)
print(f"Env created! Action space: {env.action_space.shape}")
print("Running random actions for 200 steps...")

for i in range(200):
    # Small random EE deltas + zero base velocity
    action = np.zeros(10, dtype=np.float32)
    action[:3] = np.random.uniform(-0.01, 0.01, 3)  # small xyz deltas
    action[6] = 1.0  # gripper open
    obs, reward, terminated, truncated, info = env.step(
        torch.tensor(action, dtype=torch.float32).unsqueeze(0)
    )
    env.render()
    if i % 50 == 0:
        print(f"  Step {i}, reward={reward.item():.3f}")

print("Done! Closing env.")
env.close()
