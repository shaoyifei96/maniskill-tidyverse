#!/usr/bin/env python3
"""Debug script: print base pose and controller target every few steps, save video."""
import sys, os, numpy as np, torch, cv2
import maniskill_tidyverse.tidyverse_agent
import mani_skill.envs
import gymnasium as gym

env = gym.make('RoboCasaKitchen-v1', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body',
               render_mode='rgb_array')
env.reset(seed=0)

robot = env.unwrapped.agent.robot
ctrl = env.unwrapped.agent.controller

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])

# No command — just hold rest qpos
qpos0 = robot.get_qpos().cpu().numpy()[0]
base_qpos0 = qpos0[:3]
act = torch.tensor(
    np.concatenate([ARM_HOME, [0.0], base_qpos0]),
    dtype=torch.float32).unsqueeze(0)

print(f"root pose (set_pose):  {robot.pose.p[0].cpu().numpy()}")
print(f"initial base qpos:     {base_qpos0}")
print(f"action base_cmd:       {base_qpos0}")
print(f"controller base target:{ctrl.controllers['base']._target_qpos.cpu().numpy()[0]}")
print()
print(f"{'step':>5}  {'base_qpos':>30}  {'base_link_world':>30}  {'ctrl_target':>30}")
print("-" * 105)

# Video setup
video_path = os.path.expanduser('~/tidyverse_videos/base_debug.mp4')
os.makedirs(os.path.dirname(video_path), exist_ok=True)
writer = None

for i in range(100):
    env.step(act)
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    frame = frame.astype(np.uint8)

    # Burn text overlay
    qpos = robot.get_qpos().cpu().numpy()[0]
    bl = [l for l in robot.get_links() if l.get_name() == 'base_link'][0]
    bl_pos = bl.pose.p[0].cpu().numpy()
    tgt = ctrl.controllers['base']._target_qpos.cpu().numpy()[0]

    lines = [
        f"step {i+1}",
        f"base_qpos: [{qpos[0]:.4f}, {qpos[1]:.4f}, {qpos[2]:.4f}]",
        f"base_world: [{bl_pos[0]:.2f}, {bl_pos[1]:.2f}, {bl_pos[2]:.2f}]",
        f"ctrl_target: [{tgt[0]:.4f}, {tgt[1]:.4f}, {tgt[2]:.4f}]",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for li, text in enumerate(lines):
        y = 25 + li * 22
        cv2.putText(frame, text, (10, y), font, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (10, y), font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    if i % 5 == 0 or i < 5:
        print(f"{i+1:5d}  {str(qpos[:3]):>30}  {str(bl_pos):>30}  {str(tgt):>30}")

writer.release()
print(f"\nVideo saved: {video_path}")
env.close()
