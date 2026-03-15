# ManiSkill TidyVerse Robot

TidyVerse robot agent for [ManiSkill3](https://github.com/haosulab/ManiSkill) — a Franka Panda arm on a TidyBot mobile base with a Robotiq 85 gripper. Matches the real [TidyBot](https://tidybot.cs.princeton.edu/) hardware.

## Robot Specs

- **Arm:** Franka Panda 7-DOF
- **Gripper:** Robotiq 2F-85 (parallel jaw)
- **Base:** 3-DOF mobile base (x, y, yaw)
- **Total active joints:** 16 (3 base + 7 arm + 6 gripper)
- **EE link:** `eef`

## Install

```bash
# Recommended: use a conda environment
conda create -n maniskill python=3.11 -y
conda activate maniskill
pip install mani_skill==3.0.0b22 mplib==0.2.1 pycollada
```

## Setup

```bash
git clone https://github.com/shaoyifei96/maniskill-tidyverse
cd maniskill-tidyverse

# Create symlinks (required — URDF mesh paths are relative)
ln -sf $(python3 -c "import mani_skill; print(mani_skill.__path__[0])")/assets/robots/panda/franka_description franka_description
ln -sf ~/.maniskill/data/robots/robotiq_2f/meshes robotiq_meshes
```

## Quick Start

### Run the demo script

```bash
# Set DISPLAY if running on a headless machine with Xvfb/VNC
export DISPLAY=:1  # adjust as needed

# Using conda env
conda activate maniskill
cd maniskill-tidyverse
python demo_pickcube.py
```

### Or use from Python

```python
import sys
sys.path.insert(0, '/path/to/maniskill-tidyverse')
import tidyverse_agent  # registers 'tidyverse' robot via @register_agent()
import mani_skill.envs
import gymnasium as gym

# With GUI
env = gym.make('PickCube-v1', render_mode='human', num_envs=1,
               robot_uids='tidyverse', control_mode='pd_ee_delta_pose')
obs, info = env.reset(seed=0)
```

Works with any ManiSkill3 environment (`PickCube-v1`, `RoboCasaKitchen-v1`, etc.).

> **Note:** The warning `"tidyverse is not in the task's list of supported robots"` is expected and safe to ignore.

## Control Modes

All modes use action order: **`[arm, gripper, base]`**

| Mode | Dims | Action Format |
|------|------|---------------|
| `pd_ee_delta_pose` | 10 | `[dx,dy,dz, dax,day,daz, gripper(1), base_vx,base_vy,base_vyaw]` |
| `pd_ee_pose` | 10 | `[x,y,z, ax,ay,az, gripper(1), base_vx,base_vy,base_vyaw]` |
| `pd_joint_pos` | 11 | `[arm_j1-j7(7), gripper(1), base_vx,base_vy,base_vyaw(3)]` |
| `pd_joint_delta_pos` | 11 | `[Δarm_j1-Δj7(7), Δgripper(1), base_vx,base_vy,base_vyaw(3)]` |
| **`whole_body`** | 11 | `[arm_j1-j7(7), gripper(1), base_x,base_y,base_yaw(3)]` |

> **Note:** In `pd_joint_pos` and `pd_joint_delta_pos`, the base uses **velocity** control. In `whole_body`, the base uses **position** control — required for motion planning.

## EEF Orientation Convention

The planner takes target poses as `MPPose(p=position, q=quaternion_wxyz)`.

Orientations are specified in the **world frame** (with `wrt_world=True`, the default).

**World frame axes:** X = forward, Y = left, Z = up

**EEF link frame:** The `eef` link's local Z-axis points along the gripper approach direction (the direction the fingers point). At identity quaternion `[1,0,0,0]`, the gripper points **up**.

### Euler XYZ to Quaternion

```python
from scipy.spatial.transform import Rotation as R

rot = R.from_euler('xyz', [rx, ry, rz])  # radians
q_xyzw = rot.as_quat()
q_wxyz = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]  # mplib uses wxyz
target = MPPose(p=position, q=q_wxyz)
```

### Axis Effects (rotation applied to EEF Z-axis)

| Axis | Effect | Formula |
|------|--------|---------|
| **Ry(θ)** | Tilts gripper toward **+X (forward)** | Z → `[sinθ, 0, cosθ]` |
| **Rx(θ)** | Tilts gripper toward **-Y (right)** | Z → `[0, -sinθ, cosθ]` |
| **Rz(θ)** | Spins gripper in place (yaw) | Z unchanged |

### Typical Grasp Orientations

All common grasps use **Ry rotation** to control the approach angle:

| Grasp Type | Euler XYZ (deg) | Euler XYZ (rad) | Quaternion (wxyz) | Description |
|------------|----------------|-----------------|-------------------|-------------|
| **Top-down** | `[0, 180, 0]` | `[0, π, 0]` | `[0, 0, 0, -1]` | Gripper straight down, fingers along X |
| **45° angled** | `[0, 135, 0]` | `[0, 3π/4, 0]` | `[0.383, 0, 0.924, 0]` | 45° between down and forward |
| **Forward** | `[0, 90, 0]` | `[0, π/2, 0]` | `[0.707, 0, 0.707, 0]` | Gripper horizontal, pointing forward |

To rotate the finger orientation (yaw) around the approach axis, add **Rz**:

| Variant | Euler XYZ (deg) | Description |
|---------|----------------|-------------|
| Top-down + yaw 90° | `[0, 180, 90]` | Down, fingers along Y |
| Forward + yaw 45° | `[0, 90, 45]` | Forward, fingers rotated 45° |

### Code Example

```python
import numpy as np
from scipy.spatial.transform import Rotation as R
from mplib import Pose as MPPose

def grasp_pose(position, approach='top_down', yaw_deg=0):
    """Create a grasp pose with common orientations.
    
    Args:
        position: [x, y, z] target position
        approach: 'top_down', 'angled_45', or 'forward'
        yaw_deg: rotation around approach axis (finger orientation)
    """
    ry = {'top_down': np.pi, 'angled_45': 3*np.pi/4, 'forward': np.pi/2}[approach]
    rz = np.radians(yaw_deg)
    rot = R.from_euler('xyz', [0, ry, rz])
    q = rot.as_quat()  # xyzw
    return MPPose(p=np.array(position), q=[q[3], q[0], q[1], q[2]])  # wxyz

# Examples
top_down   = grasp_pose([0.5, 0, 0.1], 'top_down')
angled     = grasp_pose([0.5, 0, 0.1], 'angled_45')
forward    = grasp_pose([0.5, 0, 0.1], 'forward')
top_yaw90  = grasp_pose([0.5, 0, 0.1], 'top_down', yaw_deg=90)
```

## Motion Planning (mplib 0.2.1)

Full whole-body 10-DOF motion planning (3 base + 7 arm) with sub-millimeter accuracy.

### Whole-Body Planning (base + arm)

```python
import mplib
from mplib.pymp import Pose as MPPose

# Use whole_body mode — base is position-controlled
env = gym.make('PickCube-v1', render_mode='human', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body')
obs, info = env.reset(seed=0)
agent = env.unwrapped.agent
robot = agent.robot

# Planning URDF: box collisions, no visual meshes (fast loading)
planner = mplib.Planner(
    urdf='tidyverse_bare.urdf',
    srdf='tidyverse_bare_mplib.srdf',  # auto-generated on first run
    move_group='eef',
)

# CRITICAL: sync mplib frame with ManiSkill robot placement
root_p = robot.pose.p[0].cpu().numpy()
root_q = robot.pose.q[0].cpu().numpy()
planner.robot.set_base_pose(MPPose(p=root_p, q=root_q))

# Plan
qpos = robot.get_qpos().cpu().numpy()[0]
target_pose = sapien.Pose(p=[0.5, 0.3, 1.0], q=[1, 0, 0, 0])
result = planner.plan_pose(target_pose, qpos, time_step=env.unwrapped.control_timestep)

if result['status'] == 'Success':
    traj = result['position']  # (N, 10): [base_x, base_y, base_yaw, j1-j7]
    gripper_val = qpos[10]
    for i in range(traj.shape[0]):
        # Map: mplib [base(3), arm(7)] → action [arm(7), gripper(1), base(3)]
        action = np.concatenate([traj[i, 3:10], [gripper_val], traj[i, 0:3]])
        env.step(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
```

### Arm-Only Planning (fixed base)

```python
# Use the arm-only planning URDF (base joints fixed)
planner = mplib.Planner(
    urdf='tidyverse_arm_planning.urdf',
    srdf='tidyverse_arm_planning_mplib.srdf',
    move_group='eef',
)

# Set base pose to base_link world position
links = {l.get_name(): l for l in robot.get_links()}
planner.robot.set_base_pose(MPPose(
    p=links['base_link'].pose.p[0].cpu().numpy(),
    q=links['base_link'].pose.q[0].cpu().numpy()))

# Planner expects [arm(7), gripper(6)] = 13 joints
arm_qpos = qpos[3:10]
gripper_qpos = qpos[10:]
planner_qpos = np.concatenate([arm_qpos, gripper_qpos])

result = planner.plan_pose(target_pose, planner_qpos, time_step=env.unwrapped.control_timestep)
# traj shape: (N, 7) — arm joints only
# Action: [arm(7), gripper(1), base_vel=0,0,0]
```

### Accuracy Results

| Mode | Error |
|------|-------|
| Arm-only | < 1mm |
| Whole-body (small reach) | 0.0mm |
| Whole-body (50cm reach) | 0.1mm |
| Whole-body (return) | 0.2mm |

## Planning URDFs

The main `tidyverse.urdf` uses DAE meshes that cause mplib to hang. Use these planning-specific URDFs instead:

| File | Use Case |
|------|----------|
| `tidyverse_bare.urdf` | Whole-body planning — box collisions, base joints active |
| `tidyverse_arm_planning.urdf` | Arm-only planning — base joints fixed |
| `tidyverse_bare_mplib.srdf` | Auto-generated SRDF for whole-body |
| `tidyverse_arm_planning_mplib.srdf` | Auto-generated SRDF for arm-only |

## RoboCasa Kitchens

120 kitchen configurations (10 layouts × 12 styles) via `RoboCasaKitchen-v1`:

```python
env = gym.make('RoboCasaKitchen-v1', num_envs=1,
               robot_uids='tidyverse', control_mode='whole_body')
obs, info = env.reset(seed=42)
```

153 object categories available. See [object spawning example](https://github.com/shaoyifei96/maniskill-tidyverse/wiki) for placing objects on counter surfaces.

## Known Limitations

- `RoboCasaKitchen-v1` is a scene viewer — no task definitions or `_check_success()`
- Fixture interaction (`is_open()`/`set_door_state()`) are stubs
- ManiSkill warns `"tidyverse is not in the task's list of supported robots"` — safe to ignore
- Whole-body planner may choose excessive base yaw rotations (cosmetic — TCP accuracy unaffected)
- mplib hangs on DAE collision meshes — always use `tidyverse_bare.urdf` for planning

## File Structure

```
maniskill-tidyverse/
├── tidyverse_agent.py              # Agent class, registered as 'tidyverse'
├── tidyverse.urdf                   # Full URDF (for ManiSkill rendering)
├── tidyverse_bare.urdf              # Planning URDF (box collisions, whole-body)
├── tidyverse_arm_planning.urdf      # Planning URDF (base fixed, arm-only)
├── tidyverse_bare_mplib.srdf        # Auto-generated SRDF
├── tidyverse_arm_planning_mplib.srdf
├── tidyverse_base/                  # Base meshes
├── franka_description/              # Symlink → Panda meshes
├── robotiq_meshes/                  # Symlink → Robotiq meshes
└── README.md
```

## License

MIT
