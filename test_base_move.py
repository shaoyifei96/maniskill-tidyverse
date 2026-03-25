#!/usr/bin/env python3
"""Test: command the base to move in x, y, yaw directions and observe."""
import sys, os, numpy as np, torch, cv2
import maniskill_tidyverse.tidyverse_agent
import mani_skill.envs
import gymnasium as gym

env = gym.make('RoboCasaKitchen-v1', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body',
               render_mode='rgb_array')
env.reset(seed=0)

robot = env.unwrapped.agent.robot

# Camera: higher + tilted down
import sapien
from mani_skill.utils import sapien_utils as _su
_rpos = robot.pose.p[0].cpu().numpy()
_cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]
_cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]
_cam_pose = _su.look_at(_cam_eye, _cam_target)
_p = _cam_pose.raw_pose[0].cpu().numpy()
_sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
for cam in env.unwrapped._human_render_cameras.values():
    cam.camera.set_local_pose(_sapien_pose)

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
SETTLE_THRESH = 0.005  # 5mm / 5mrad — close enough

def get_qpos():
    return robot.get_qpos().cpu().numpy()[0]

def make_action(base_cmd):
    act = np.concatenate([ARM_HOME, [0.0], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)

# Video setup
video_path = os.path.expanduser('~/tidyverse_videos/base_move.mp4')
os.makedirs(os.path.dirname(video_path), exist_ok=True)
writer = None

def step_and_record(action, label):
    global writer
    env.step(action)
    frame = env.render()
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
    if frame.ndim == 4:
        frame = frame[0]
    frame = frame.astype(np.uint8).copy()

    q = get_qpos()
    bl = [l for l in robot.get_links() if l.get_name() == 'base_link'][0]
    bl_pos = bl.pose.p[0].cpu().numpy()
    lines = [
        label,
        f"base_qpos: [{q[0]:.3f}, {q[1]:.3f}, {q[2]:.3f}]",
        f"base_world: [{bl_pos[0]:.2f}, {bl_pos[1]:.2f}, {bl_pos[2]:.2f}]",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for li, text in enumerate(lines):
        y = 20 + li * 18
        cv2.putText(frame, text, (8, y), font, 0.45, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, text, (8, y), font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(video_path, fourcc, 30, (w, h))
    writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def move_to(target, label, max_steps=80):
    """Command base to target, stop early when within SETTLE_THRESH."""
    action = make_action(target)
    for i in range(max_steps):
        step_and_record(action, f"{label} ({i+1})")
        err = np.max(np.abs(get_qpos()[:3] - target))
        if err < SETTLE_THRESH:
            print(f"  {label}: settled at step {i+1}, err={err:.5f}")
            return i + 1
    print(f"  {label}: max steps reached, err={np.max(np.abs(get_qpos()[:3] - target)):.5f}")
    return max_steps


q0 = get_qpos()[:3].copy()
print(f"Initial base qpos: {q0}")
print(f"Initial base world: {robot.pose.p[0].cpu().numpy()}")

# Hold still briefly
move_to(q0, "Hold", max_steps=20)

# +X 0.3m
target_x = q0.copy(); target_x[0] += 0.3
move_to(target_x, "+X 0.3m")

# +Y 0.3m
target_y = target_x.copy(); target_y[1] += 0.3
move_to(target_y, "+Y 0.3m")

# +yaw 0.5 rad
target_yaw = target_y.copy(); target_yaw[2] += 0.5
move_to(target_yaw, "+yaw 0.5rad")

# Return to origin
move_to(q0, "Return", max_steps=120)

writer.release()
print(f"\nVideo saved: {video_path}")
env.close()
