"""Test arm-only IK with different orientations at the starting TCP position.

Usage:
    DISPLAY=:1 python3 test_orientations.py
"""
import sys, numpy as np
sys.path.insert(0, '/home/yifei/.openclaw/workspace/maniskill_tidyverse')
import tidyverse_agent, mani_skill.envs
import gymnasium as gym
import sapien, mplib, torch
from mplib.pymp import Pose as MPPose
from transforms3d.euler import euler2quat, quat2euler
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import sapien_utils
from mani_skill.sensors.camera import CameraConfig

bare_urdf = '/home/yifei/.openclaw/workspace/maniskill_tidyverse/tidyverse_bare.urdf'
bare_srdf = '/home/yifei/.openclaw/workspace/maniskill_tidyverse/tidyverse_bare_mplib.srdf'
out_dir = '/home/yifei/.openclaw/workspace/maniskill_tidyverse/videos'

# Zoom out camera
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
@property
def _zoomed_cam(self):
    pose = sapien_utils.look_at(eye=[1.2, 1.4, 1.2], target=[-0.1, 0.0, 0.3])
    return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)
PickCubeEnv._default_human_render_camera_configs = _zoomed_cam

env = gym.make('PickCube-v1', num_envs=1, robot_uids='tidyverse',
               control_mode='whole_body', render_mode='rgb_array')
env = RecordEpisode(env, output_dir=out_dir, save_video=True, max_steps_per_video=4000, video_fps=30)

obs, _ = env.reset(seed=0)
uw = env.unwrapped
root_p = uw.agent.robot.pose.p[0].cpu().numpy()
root_q = uw.agent.robot.pose.q[0].cpu().numpy()
tcp_p = uw.agent.tcp_pose.p[0].cpu().numpy()

planner = mplib.Planner(urdf=bare_urdf, srdf=bare_srdf, move_group='eef')
planner.robot.set_base_pose(MPPose(p=root_p, q=root_q))

# Mask for move_group joints (10 = 3 base + 7 arm). True = locked.
arm_only_mask = [True]*3 + [False]*7  # lock base, free arm

def make_action(traj_wp):
    """whole_body action = [base(3), arm(7), gripper(1)]"""
    action = np.zeros(11)
    action[:10] = traj_wp[:10]
    action[10] = 1.0  # gripper open
    return torch.tensor(action[np.newaxis], dtype=torch.float32)

print(f"TCP start: {np.round(tcp_p, 3)}")
print(f"Move group joints: {planner.user_joint_names[:10]}")

orientations = {
    'top_down':       (np.pi, 0, 0),
    'tilted_fwd_30':  (np.pi*5/6, 0, 0),
    'tilted_fwd_45':  (np.pi*3/4, 0, 0),
    'rot_z_45':       (np.pi, 0, np.pi/4),
    'rot_z_90':       (np.pi, 0, np.pi/2),
    'tilted_side_30': (np.pi, np.pi/6, 0),
}

for name, (rx, ry, rz) in orientations.items():
    q = np.array(euler2quat(rx, ry, rz, 'sxyz'))
    qf = uw.agent.robot.get_qpos().cpu().numpy()[0]

    r = planner.plan_pose(sapien.Pose(p=tcp_p, q=q), qf, mask=arm_only_mask,
                          time_step=uw.control_timestep)
    deg = f"({np.degrees(rx):.0f},{np.degrees(ry):.0f},{np.degrees(rz):.0f})"

    if r['status'] == 'Success':
        traj = r['position']
        for wp in traj:
            obs, rew, term, trunc, info = env.step(make_action(wp))
        for _ in range(40):
            obs, rew, term, trunc, info = env.step(make_action(traj[-1]))

        ach_p = uw.agent.tcp_pose.p[0].cpu().numpy()
        ach_euler = np.degrees(quat2euler(uw.agent.tcp_pose.q[0].cpu().numpy(), 'sxyz'))
        pos_err = np.linalg.norm(ach_p - tcp_p)
        print(f"  ✅ {name:<20} target={deg:<20} euler={np.round(ach_euler,1)} pos_err={pos_err:.4f}")
    else:
        print(f"  ❌ {name:<20} target={deg:<20} {r['status']}")

env.close()

import os
for f in sorted(os.listdir(out_dir)):
    if f.endswith('.mp4'):
        print(f"  Video: {os.path.join(out_dir, f)} ({os.path.getsize(os.path.join(out_dir, f))/1e6:.1f} MB)")
