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

## Protocol Bridges (optional — for agent_server integration)

To run ManiSkill as a drop-in replacement for real hardware via [agent_server](https://github.com/TidyBot-Services), install the protocol bridge drivers from the [TidyBot-Services org](https://github.com/orgs/TidyBot-Services/repositories?q=mirror%3Afalse+fork%3Afalse+archived%3Afalse):

```bash
pip install git+https://github.com/TidyBot-Services/maniskill_server.git
pip install git+https://github.com/TidyBot-Services/arm_franka_maniskill_service.git
pip install git+https://github.com/TidyBot-Services/gripper_robotiq_maniskill_service.git
pip install git+https://github.com/TidyBot-Services/base_tidybot_maniskill_service.git
pip install git+https://github.com/TidyBot-Services/camera_realsense_maniskill_service.git
```

Then start the sim server: `python -m maniskill_server --gui`

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

![Grasp Orientations](orientation_images/grasp_orientations_grid.png)

| Grasp Type | Euler XYZ (deg) | Euler XYZ (rad) | Quaternion (wxyz) | Description |
|------------|----------------|-----------------|-------------------|-------------|
| **Top-down** | `[0, 180, 0]` | `[0, π, 0]` | `[0, 0, 0, -1]` | Gripper straight down, fingers along X |
| **45° angled** | `[0, 135, 0]` | `[0, 3π/4, 0]` | `[0.383, 0, 0.924, 0]` | 45° between down and forward |
| **Forward** | `[0, 90, 0]` | `[0, π/2, 0]` | `[0.707, 0, 0.707, 0]` | Gripper horizontal, pointing forward |

To rotate the finger orientation (yaw) around the approach axis, add **Rz**:

| Variant | Euler XYZ (deg) | Description |
|---------|----------------|-------------|
| Top-down + yaw 90° | `[0, 180, 90]` | Down, fingers along Y (bottom-right in grid) |
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

## Table Grasp Test

`test_table_grasp.py` — end-to-end pick test with a table and a small red block. The robot plans pre-grasp, approach, close gripper, and lift for four grasp strategies:

| Strategy | Best For | Description |
|----------|----------|-------------|
| **Top-down** | Objects on surfaces | Gripper straight down |
| **Front** | Vertical handles (e.g. cabinet doors) | Gripper horizontal, pointing forward. Not ideal for objects sitting on a surface — the table blocks the approach. |
| **Front-Vertical** | Horizontal handles (e.g. drawer pulls) | Same as Front but fingers rotated 90° so they're vertical, wrapping around a horizontal bar |
| **Angled 45°** | General purpose | 45° between down and forward |

```bash
# GUI mode
python test_table_grasp.py --render human

# Record video with text overlays (saved to videos/)
python test_table_grasp.py --render rgb_array

# Options
python test_table_grasp.py --robot-x -0.3 --table-x 0.0 --table-height 0.762
```

Uses `SapienPlanner` / `SapienPlanningWorld` for collision-aware motion planning with automatic obstacle detection.

### Planning Strategy

1. **Grasp IK** — solve IK for the grasp pose first (arm-only, then whole-body fallback)
2. **Pre-grasp IK** — solve IK for the pre-grasp pose (8cm above), seeded from the grasp IK solution so the arm config is already close to what approach needs
3. **Pre-grasp path** — plan joint-space path from current config to the pre-grasp IK solution
4. **Approach** — plan from pre-grasp down to the grasp pose (arm-only if grasp IK was arm-only, otherwise whole-body)
5. **Close gripper** → **Lift** → **Open gripper** → **Return to home**

This IK-seeded approach ensures the pre-grasp arm configuration is close to the grasp configuration, so the approach step is a short, smooth motion rather than a large reconfiguration.

### Features

- **Arm-only with whole-body fallback**: tries arm-only first (base stays fixed), falls back to whole-body (base moves) if needed
- **Base locking**: when using arm-only planning, base commands from the trajectory are clamped to prevent RRT drift
- **Settle check**: waits for both arm and base joints to converge after each trajectory
- **Video recording**: text overlays showing current stage, 1-second pause between stages
- **Failure diagnostics**: IK feasibility, obstacle list, collision pairs, joint limit warnings
- **No physics resets**: drops the block and plans back to home between grasps instead of teleporting

### Gripper Control

The Robotiq 85 gripper joint range is `[0, 0.81]` rad in `whole_body` mode (`normalize_action=False`):
- **0.0** = fully open
- **0.81** = fully closed

### Base PD Gains

Base uses critically damped PD control (Kp=1000, Kd=520) for fast settling without oscillation.

## RoboCasa Kitchen Grasp Test

`test_robocasa_grasp.py` — places a cube on every available surface in a RoboCasa kitchen, then attempts to pick each one up using the IK-seeded grasp pipeline.

```bash
# GUI mode
python test_robocasa_grasp.py --render human --seed 0

# Record video (saved to videos/)
python test_robocasa_grasp.py --render rgb_array --seed 0

# Limit to N nearest cubes
python test_robocasa_grasp.py --render rgb_array --seed 0 --max-cubes 5
```

### Placement surfaces

The script enumerates all fixtures and finds placement locations on:
- **Counters** — every top geom region
- **Stoves/Stovetops** — each burner site
- **Cabinets** (Single/Hinge/Open) — interior bottom shelf + top surface
- **Drawers** — interior (if at reachable height)
- **Microwaves** — interior chamber
- **Coffee machines** — receptacle site
- **Sinks** — basin interior

Typically finds 40+ placement locations per kitchen layout.

### Grasp strategy

- **Open surfaces** (counters, stove, sink): Top-Down → Angled 45° → Front
- **Enclosed spaces** (cabinet/drawer/microwave interiors): Front → Angled 45°
- Cubes are sorted by distance from the arm base; nearest attempted first
- Arm-only planning first, whole-body fallback if out of reach
- Timeout guards (`signal.alarm`) prevent infinite planning hangs

### Collision handling caveat

This older script relaxes **all** kitchen fixtures in the ACM so the planner ignores furniture entirely. See `test_perception_grasp.py` for the improved approach with fixture AABB boxes and dual-radius ACM.

## Perception-Based Grasp Pipeline

`test_perception_grasp.py` — uses depth + segmentation cameras to detect objects, then plans and executes grasps with wrist-camera refinement. Modular: split across `perception.py`, `grasp_strategies.py`, `planning_utils.py`, and `execution.py`.

```bash
# Headless — record video
python test_perception_grasp.py --render rgb_array --seed 0 --spawn-test-objects

# With obstacle map snapshots (saved as .glb)
python test_perception_grasp.py --render rgb_array --seed 0 --spawn-test-objects --viz-dir perception_viz

# Limit objects
python test_perception_grasp.py --render rgb_array --seed 0 --spawn-test-objects --max-objects 3
```

### Pipeline stages

1. **Perceive** — base camera RGB+depth+segmentation → back-project to 3D world frame → bbox midpoint center estimation (~5mm accuracy vs ground truth)
2. **Strategy selection** — TopDown + Angled45, each at 3 yaw angles (base→object direction, ±30°) = 6 candidates per object
3. **Pre-grasp IK** — solve IK for 8cm above target; tries arm-only then whole-body, with 180° gripper flip fallback
4. **Plan to pre-grasp** — whole-body collision-aware path via mplib RRT
5. **Base tracking check** — verify base reached planned position (< 10cm error), skip if stuck
6. **Wrist re-perception** — re-detect target with wrist camera at pre-grasp position (~30mm refinement)
7. **Approach** — plan to refined grasp pose
8. **Grasp → Lift → Transport → Drop → Return home**

### Obstacle handling

Kitchen fixtures are approximated as **AABB boxes** added to the mplib planning world (FCL objects). These boxes are planning-only — they do not affect SAPIEN physics. The ACM uses a **dual-radius** scheme: objects within 1.5m of the robot start position OR any target position are collision-checked; distant objects are relaxed.

> **Caveat: obstacles are approximate.** The fixture AABB boxes are axis-aligned bounding boxes computed from collision meshes. They over-approximate curved/angled fixtures and may block valid paths near tight spaces. Articulated fixture meshes (cabinet doors, drawers) are relaxed in the ACM due to false-positive collisions — only the AABB boxes represent them. This means the planner does not account for open/closed door states. Proper collision handling would require convex decomposition or using the actual fixture mesh state.

### Modules

| File | Contents |
|------|----------|
| `perception.py` | `PerceptionResult`, `deproject_pixels_to_world`, `perceive_objects`, `classify_fixture_context`, `save_perception_debug` |
| `grasp_strategies.py` | Grasp pose builders (TopDown, Angled45, Front), `choose_grasp_strategy` with yaw variations |
| `planning_utils.py` | Monkey-patch for Robotiq meshes, AABB computation, `add_fixture_boxes_to_planner`, `build_kitchen_acm`, `resolve_start_collisions` |
| `execution.py` | Constants, `make_action`, `execute_trajectory`, `attempt_grasp` (full pipeline), 180° flip IK fallback |
| `viz_planning_world.py` | Export mplib planning world collision meshes to .glb for inspection |

## Known Limitations

- `RoboCasaKitchen-v1` is a scene viewer — no task definitions or `_check_success()`
- Fixture interaction (`is_open()`/`set_door_state()`) are stubs
- ManiSkill warns `"tidyverse is not in the task's list of supported robots"` — safe to ignore
- Whole-body planner may choose excessive base yaw rotations (cosmetic — TCP accuracy unaffected)
- mplib hangs on DAE collision meshes — always use `tidyverse_bare.urdf` for planning
- **Obstacle approximation:** fixture AABB boxes are coarse — they over-approximate geometry and don't reflect articulated state (open/closed doors). See Perception Pipeline section above.

## File Structure

```
maniskill-tidyverse/
├── test_perception_grasp.py         # Perception-based grasp pipeline (main script)
├── perception.py                    # Camera perception: depth back-projection, object detection
├── grasp_strategies.py              # Grasp pose generation and strategy selection
├── planning_utils.py                # mplib monkey-patch, AABB boxes, ACM builder
├── execution.py                     # Motion execution, attempt_grasp pipeline
├── viz_planning_world.py            # Export planning world to .glb
├── test_robocasa_grasp.py           # Kitchen grasp test (place cubes, pick them up)
├── test_robocasa_pick_place.py      # Pick-and-place with known object positions
├── test_table_grasp.py              # Table-top grasp test (3 strategies, video)
├── debug_perception_offset.py       # Compare ground-truth vs perceived positions
├── tidyverse_agent.py               # Agent class, registered as 'tidyverse'
├── tidyverse.urdf                   # Full URDF (ManiSkill rendering)
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
