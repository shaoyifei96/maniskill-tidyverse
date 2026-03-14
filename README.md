# ManiSkill TidyVerse Agent

TidyVerse robot (Panda arm + mobile base + Robotiq 85 gripper) ported to ManiSkill3.

## Structure
- `tidyverse.urdf` — Full URDF: 3-DOF base + 7-DOF Panda arm + Robotiq 85 gripper
- `tidyverse_agent.py` — ManiSkill3 agent class with controllers
- `tidyverse_base/` — Base mesh files (STL)

## Dependencies
Requires ManiSkill3 with the following assets:
- Panda: `franka_description` (ships with ManiSkill)
- Robotiq: `python -m mani_skill.utils.download_asset robotiq_2f`

## Usage
```python
import sys
sys.path.insert(0, '/path/to/maniskill_tidyverse')
import tidyverse_agent  # registers the agent

import gymnasium as gym
env = gym.make('RoboCasaKitchen-v1', render_mode='rgb_array', robot_uids='tidyverse')
```

## Action Space (11-DOF)
- Base: 3 (x velocity, y velocity, yaw velocity)
- Arm: 7 (Panda joint positions/deltas)
- Gripper: 1 (Robotiq open/close)
