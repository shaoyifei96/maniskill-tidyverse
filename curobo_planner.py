"""cuRobo-based motion planner for Franka Panda on mobile base.

Replaces mplib SapienPlanner with GPU-accelerated collision-aware planning.
Uses cuRobo's built-in franka_mobile config (base_x, base_y, base_z/yaw + 7-DOF arm).

The planner operates in world frame. Collision objects (kitchen fixtures) are
added as cuboids. cuRobo handles collision checking on GPU.
"""

import numpy as np
import torch
import time
from typing import Optional


class CuroboPlanner:
    """GPU-accelerated motion planner using NVIDIA cuRobo."""

    # Lock values for the gripper fingers (match franka_tidyverse.yml lock_joints)
    _FINGER_LOCKS = {"panda_finger_joint1": 0.04, "panda_finger_joint2": 0.04}
    _ROBOT_CFG_YML = "franka_tidyverse.yml"

    def __init__(self, device: str = "cuda:0"):
        self._device = torch.device(device)
        self._motion_gen = None       # whole-body (base free)
        self._motion_gen_arm = None   # arm-only (base_x/y/z locked)
        self._arm_only_cfg_dict = None
        self._warmed_up = False
        self._world_cuboids = []
        self._last_arm_base_lock = None  # (x, y, z) of last update_locked_joints call

    def warmup(self):
        """One-time initialization: load robot config + CUDA JIT compile.

        Takes ~30s on first call due to CUDA kernel compilation.
        """
        if self._warmed_up:
            return

        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig
        from curobo.geom.types import WorldConfig, Cuboid
        from curobo.types.base import TensorDeviceType

        tensor_args = TensorDeviceType(device=self._device)

        t0 = time.time()
        print("[curobo] Loading franka_mobile config...")

        # cuRobo requires at least one collision object for warmup
        dummy_world = WorldConfig(cuboid=[
            Cuboid(name="ground", pose=[0, 0, -0.5, 1, 0, 0, 0], dims=[10, 10, 0.01])
        ])
        mg_config = MotionGenConfig.load_from_robot_config(
            robot_cfg=self._ROBOT_CFG_YML,
            world_model=dummy_world,
            tensor_args=tensor_args,
            interpolation_dt=0.05,
            collision_cache={"obb": 100, "mesh": 10},  # large cache for kitchen fixtures
            collision_activation_distance=0.01,  # 10mm collision margin
            use_cuda_graph=False,  # allow switching between arm_only/whole_body
        )
        self._motion_gen = MotionGen(mg_config)

        print("[curobo] Warming up CUDA kernels...")
        self._motion_gen.warmup()

        # Load a second MotionGen with base joints locked, for lock_base=True mode.
        # update_locked_joints requires the lock set to have the same cardinality as
        # what was provided at init, so we initialize the full 5-joint lock set here
        # (gripper fingers + base_x/y/z) and only vary the VALUES per plan call.
        print("[curobo] Loading arm-only config (base joints locked)...")
        from curobo.util_file import get_robot_configs_path, join_path, load_yaml
        cfg_dict = load_yaml(join_path(get_robot_configs_path(), self._ROBOT_CFG_YML))
        arm_locks = dict(self._FINGER_LOCKS)
        arm_locks.update({"base_x": 0.0, "base_y": 0.0, "base_z": 0.0})
        cfg_dict["robot_cfg"]["kinematics"]["lock_joints"] = arm_locks
        self._arm_only_cfg_dict = cfg_dict
        mg_config_arm = MotionGenConfig.load_from_robot_config(
            robot_cfg=cfg_dict["robot_cfg"],
            world_model=dummy_world,
            tensor_args=tensor_args,
            interpolation_dt=0.05,
            collision_cache={"obb": 100, "mesh": 10},
            collision_activation_distance=0.01,
            use_cuda_graph=False,
        )
        self._motion_gen_arm = MotionGen(mg_config_arm)
        self._motion_gen_arm.warmup()

        dt = time.time() - t0
        print(f"[curobo] Ready ({dt:.1f}s, whole_body + arm_only)")
        self._warmed_up = True

    def set_collision_world(self, cuboids: list[dict], robot_pos: Optional[np.ndarray] = None,
                            max_distance: float = 3.0):
        """Update collision world with kitchen fixture cuboids.

        Args:
            cuboids: list of {"name": str, "center": [x,y,z], "half_size": [hx,hy,hz]}
            robot_pos: [x, y] robot base position — cuboids overlapping this are skipped
            max_distance: only include cuboids within this distance from robot (meters)
        """
        from curobo.geom.types import WorldConfig, Cuboid as CuCuboid

        self._world_cuboids = cuboids

        cu_cuboids = []
        skipped_overlap = 0
        skipped_far = 0
        for c in cuboids:
            cx, cy, cz = c["center"]
            hx, hy, hz = c["half_size"]

            if robot_pos is not None:
                rx, ry = robot_pos[0], robot_pos[1]

                # Skip cuboids that overlap with robot position (cause start-state collision)
                margin = 0.15  # 15cm margin
                if (cx - hx - margin < rx < cx + hx + margin and
                    cy - hy - margin < ry < cy + hy + margin):
                    skipped_overlap += 1
                    continue

                # Skip cuboids too far from robot (reduce optimization burden)
                dist = np.sqrt((cx - rx)**2 + (cy - ry)**2)
                if dist > max_distance:
                    skipped_far += 1
                    continue

            cu_cuboids.append(CuCuboid(
                name=c["name"],
                pose=[cx, cy, cz, 1, 0, 0, 0],  # position + identity quaternion
                dims=[hx * 2, hy * 2, hz * 2],   # cuRobo uses full dims, not half
            ))

        world_config = WorldConfig(cuboid=cu_cuboids)
        self._motion_gen.update_world(world_config)
        if self._motion_gen_arm is not None:
            self._motion_gen_arm.update_world(world_config)
        print(f"[curobo] Updated collision world: {len(cu_cuboids)} cuboids "
              f"(skipped {skipped_overlap} overlap, {skipped_far} far)")

    def validate_base_path(self, base_positions: np.ndarray,
                           target_pos: Optional[np.ndarray] = None,
                           base_radius: float = 0.20,
                           target_exclusion_radius: float = 0.15,
                           base_box: Optional[dict] = None) -> tuple:
        """Validate that a base trajectory does not collide with kitchen fixtures.

        If base_box is given (keys: center_xy, half_extents), uses an oriented
        bounding box (OBB) check via SAT — base_positions[:, 2] is treated as
        world yaw of the OBB. Otherwise falls back to a circular footprint.

        Filters applied:
        - Waypoint 0 skipped (robot is there, known clear)
        - Walls/floors skipped (room boundaries that may contain robot)
        - Cuboids near the target skipped (base must approach target counters/sinks)
        - Cuboids overlapping the start footprint skipped (false positives)

        Args:
            base_positions: (T, 3) world-frame waypoints (x, y, yaw)
            target_pos: optional [x, y, z] EE target — cuboids within
                target_exclusion_radius of this XY are not collision-checked
            base_radius: mobile base footprint radius (circle mode)
            target_exclusion_radius: XY radius around target where collision is allowed
            base_box: optional dict with "center_xy" and "half_extents" describing
                the base OBB in base_link's local frame

        Returns:
            (collision_detected: bool, first_idx: int, fixture_name: str)
        """
        if not self._world_cuboids or len(base_positions) <= 1:
            return False, -1, ""

        use_obb = base_box is not None
        if use_obb:
            box_offset = np.array(base_box.get("center_xy", [0.0, 0.0]))
            box_half = np.array(base_box["half_extents"])

        # Compute start footprint to identify cuboids the robot already
        # touches at spawn (these would otherwise yield false positives).
        start_x = float(base_positions[0, 0])
        start_y = float(base_positions[0, 1])
        start_yaw = float(base_positions[0, 2]) if base_positions.shape[1] > 2 else 0.0

        if use_obb:
            cos_s, sin_s = np.cos(start_yaw), np.sin(start_yaw)
            start_ox = start_x + cos_s * box_offset[0] - sin_s * box_offset[1]
            start_oy = start_y + sin_s * box_offset[0] + cos_s * box_offset[1]
            start_x_axis = (cos_s, sin_s)
            start_y_axis = (-sin_s, cos_s)
            start_margin = 0.05  # inflate AABBs slightly so touching counts as overlap

        start_inside = set()
        for c in self._world_cuboids:
            cx, cy = c["center"][0], c["center"][1]
            hx, hy = c["half_size"][0], c["half_size"][1]
            cz, hz = c["center"][2], c["half_size"][2]
            if cz + hz < 0.0 or cz - hz > 0.5:
                continue
            if use_obb:
                hx_m = hx + start_margin
                hy_m = hy + start_margin
                dx_c = cx - start_ox
                dy_c = cy - start_oy
                overlap = True
                for ax, ay in [(1.0, 0.0), (0.0, 1.0), start_x_axis, start_y_axis]:
                    dist = abs(dx_c * ax + dy_c * ay)
                    aabb_proj = hx_m * abs(ax) + hy_m * abs(ay)
                    obb_proj = (box_half[0] * abs(start_x_axis[0] * ax + start_x_axis[1] * ay) +
                                box_half[1] * abs(start_y_axis[0] * ax + start_y_axis[1] * ay))
                    if dist > obb_proj + aabb_proj:
                        overlap = False
                        break
                if overlap:
                    start_inside.add(c["name"])
            else:
                if (cx - hx - base_radius <= start_x <= cx + hx + base_radius and
                    cy - hy - base_radius <= start_y <= cy + hy + base_radius):
                    start_inside.add(c["name"])

        # Filter out walls/floors, target-adjacent cuboids, and start-containing cuboids
        active_cuboids = []
        for c in self._world_cuboids:
            name_lower = c["name"].lower()
            if "wall" in name_lower or "floor" in name_lower:
                continue
            if c["name"] in start_inside:
                continue
            if target_pos is not None:
                cx, cy = c["center"][0], c["center"][1]
                tdx = cx - float(target_pos[0])
                tdy = cy - float(target_pos[1])
                if tdx * tdx + tdy * tdy < target_exclusion_radius ** 2:
                    continue
            active_cuboids.append(c)

        r2 = base_radius * base_radius

        # Start from waypoint 1 — waypoint 0 is the current state (known clear)
        for idx in range(1, len(base_positions)):
            x, y = float(base_positions[idx, 0]), float(base_positions[idx, 1])
            yaw = float(base_positions[idx, 2]) if base_positions.shape[1] > 2 else 0.0

            if use_obb:
                cos_y, sin_y = np.cos(yaw), np.sin(yaw)
                ox = x + cos_y * box_offset[0] - sin_y * box_offset[1]
                oy = y + sin_y * box_offset[0] + cos_y * box_offset[1]
                obb_x_axis = (cos_y, sin_y)
                obb_y_axis = (-sin_y, cos_y)

            for c in active_cuboids:
                cx, cy, cz = c["center"]
                hx, hy, hz = c["half_size"]

                # Skip fixtures whose Z extent doesn't overlap base height (0 to ~0.5m)
                if cz + hz < 0.0 or cz - hz > 0.5:
                    continue

                if use_obb:
                    # SAT: separating axis test for OBB vs AABB in 2D
                    dx_c = cx - ox
                    dy_c = cy - oy
                    overlap = True
                    for ax, ay in [(1.0, 0.0), (0.0, 1.0), obb_x_axis, obb_y_axis]:
                        dist = abs(dx_c * ax + dy_c * ay)
                        aabb_proj = hx * abs(ax) + hy * abs(ay)
                        obb_proj = (box_half[0] * abs(obb_x_axis[0] * ax + obb_x_axis[1] * ay) +
                                    box_half[1] * abs(obb_y_axis[0] * ax + obb_y_axis[1] * ay))
                        if dist > obb_proj + aabb_proj:
                            overlap = False
                            break
                    if overlap:
                        return True, idx, c["name"]
                else:
                    dx_c = max(cx - hx - x, 0.0, x - (cx + hx))
                    dy_c = max(cy - hy - y, 0.0, y - (cy + hy))
                    if dx_c * dx_c + dy_c * dy_c < r2:
                        return True, idx, c["name"]

        return False, -1, ""

    def _update_collision_for_target(self, target_pos: np.ndarray,
                                      robot_pos: Optional[np.ndarray] = None):
        """Update collision world, excluding cuboids that contain the target.

        In kitchen manipulation, the arm often needs to reach INTO fixtures
        (cabinets, sinks). Cuboids containing the target are excluded so the
        arm can enter them.
        """
        from curobo.geom.types import WorldConfig, Cuboid as CuCuboid

        if not self._world_cuboids:
            return

        margin = 0.05  # 5cm margin around target for exclusion
        cu_cuboids = []
        skipped_target = 0
        skipped_overlap = 0
        skipped_far = 0
        for c in self._world_cuboids:
            cx, cy, cz = c["center"]
            hx, hy, hz = c["half_size"]

            # Skip cuboids near the target (arm needs to reach through them)
            tx, ty, tz = target_pos
            dist_to_target = np.sqrt((cx - tx)**2 + (cy - ty)**2)
            if dist_to_target < 0.5:  # 50cm radius around target
                skipped_target += 1
                continue

            if robot_pos is not None:
                rx, ry = robot_pos[0], robot_pos[1]
                # Skip cuboids overlapping robot base
                rm = 0.15
                if (cx - hx - rm < rx < cx + hx + rm and
                    cy - hy - rm < ry < cy + hy + rm):
                    skipped_overlap += 1
                    continue
                # Skip far cuboids
                dist = np.sqrt((cx - rx)**2 + (cy - ry)**2)
                if dist > 3.0:
                    skipped_far += 1
                    continue

            cu_cuboids.append(CuCuboid(
                name=c["name"],
                pose=[cx, cy, cz, 1, 0, 0, 0],
                dims=[hx * 2, hy * 2, hz * 2],
            ))

        world_config = WorldConfig(cuboid=cu_cuboids)
        self._motion_gen.update_world(world_config)
        if self._motion_gen_arm is not None:
            self._motion_gen_arm.update_world(world_config)
        print(f"[curobo] Collision world for target: {len(cu_cuboids)} cuboids "
              f"(excl {skipped_target} at target, {skipped_overlap} overlap, {skipped_far} far)")

    def _select_mg(self, lock_base: bool, current_q: np.ndarray):
        """Return (motion_gen, q_active) for the selected mode.

        For lock_base=True, updates the arm-only MotionGen's base lock values
        to match current_q[:3] (if changed since last call) and returns the
        7-DOF arm slice of current_q. Otherwise returns the whole-body MotionGen
        and the 10-DOF slice.
        """
        if lock_base:
            if self._motion_gen_arm is None:
                raise RuntimeError("lock_base=True requires warmup() to have completed")
            base_key = (float(current_q[0]), float(current_q[1]), float(current_q[2]))
            if base_key != self._last_arm_base_lock:
                arm_locks = dict(self._FINGER_LOCKS)
                arm_locks["base_x"] = base_key[0]
                arm_locks["base_y"] = base_key[1]
                arm_locks["base_z"] = base_key[2]
                self._motion_gen_arm.update_locked_joints(
                    arm_locks, self._arm_only_cfg_dict["robot_cfg"])
                self._last_arm_base_lock = base_key
            return self._motion_gen_arm, np.asarray(current_q[3:10], dtype=np.float32)
        return self._motion_gen, np.asarray(current_q[:10], dtype=np.float32)

    def _widen(self, partial: np.ndarray, current_q: np.ndarray,
               lock_base: bool) -> np.ndarray:
        """Widen an arm-only result back to 10-DOF (fixed base prepended)."""
        if not lock_base:
            return partial
        if partial.ndim == 1:
            full = np.zeros(10, dtype=partial.dtype)
            full[:3] = current_q[:3]
            full[3:] = partial
            return full
        T = partial.shape[0]
        full = np.zeros((T, 10), dtype=partial.dtype)
        full[:, :3] = current_q[:3]
        full[:, 3:] = partial
        return full

    def plan_pose(self, current_q: np.ndarray, target_pos: np.ndarray,
                  target_quat: np.ndarray, lock_base: bool = False,
                  max_attempts: int = 10) -> Optional[np.ndarray]:
        """Plan trajectory to target EE pose.

        Args:
            current_q: Current joint state (10 values: base3 + arm7)
            target_pos: Target EE position [x, y, z] in world frame
            target_quat: Target EE orientation [w, x, y, z]
            lock_base: If True, only plan arm motion (base stays fixed at current_q[:3])
            max_attempts: Number of planning attempts

        Returns:
            (T, 10) numpy array of joint positions, or None on failure
        """
        from curobo.types.math import Pose
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        if not self._warmed_up:
            self.warmup()

        # Update collision world excluding target fixture
        robot_pos = current_q[:2] if len(current_q) >= 2 else None
        self._update_collision_for_target(target_pos, robot_pos=robot_pos)

        mg, q_active = self._select_mg(lock_base, current_q)
        joint_names = mg.robot_cfg.kinematics.cspace.joint_names

        q_tensor = torch.as_tensor(q_active, dtype=torch.float32,
                                   device=self._device).unsqueeze(0)
        current_state = JointState.from_position(q_tensor, joint_names=joint_names)

        # Build goal pose (use np.array→as_tensor to avoid the slow list-of-ndarray path)
        pos_np = np.asarray(target_pos, dtype=np.float32).reshape(1, 3)
        quat_np = np.asarray(target_quat, dtype=np.float32).reshape(1, 4)
        goal = Pose(
            position=torch.as_tensor(pos_np, device=self._device),
            quaternion=torch.as_tensor(quat_np, device=self._device),
        )

        plan_config = MotionGenPlanConfig(
            max_attempts=max_attempts,
            enable_graph=True,
            enable_opt=True,
            need_graph_success=False,
            finetune_dt_scale=0.95,
            finetune_attempts=7,
            finetune_dt_decay=1.05,
            num_trajopt_seeds=12,
            num_graph_seeds=8,
        )

        t0 = time.time()
        try:
            result = mg.plan_single(current_state, goal, plan_config)
            # If start state collision, retry with cleared collision world
            if not result.success[0] and "START_STATE" in str(result.status):
                print(f"[curobo] Start state collision, retrying with cleared world")
                from curobo.geom.types import WorldConfig as WC2, Cuboid as Cb2
                mg.update_world(WC2(cuboid=[
                    Cb2(name="ground", pose=[0,0,-1,1,0,0,0], dims=[0.01,0.01,0.01])
                ]))
                result = mg.plan_single(current_state, goal, plan_config)
        except Exception as e:
            dt = time.time() - t0
            print(f"[curobo] Plan ERROR ({dt:.2f}s): {e}")
            import traceback; traceback.print_exc()
            return None
        dt = time.time() - t0

        if result.success[0]:
            traj = result.get_interpolated_plan()
            positions = traj.position.cpu().numpy()  # (T, 7) arm-only or (T, 10) whole-body
            positions = self._widen(positions, current_q, lock_base)
            print(f"[curobo] Plan OK ({dt:.2f}s, {'arm_only' if lock_base else 'whole_body'}): "
                  f"{positions.shape[0]} waypoints")
            return positions
        else:
            print(f"[curobo] Plan FAILED ({dt:.2f}s): {result.status}")
            return None

    def plan_joints(self, current_q: np.ndarray,
                    target_q: np.ndarray,
                    lock_base: bool = False) -> Optional[np.ndarray]:
        """Plan trajectory to target joint configuration.

        Args:
            current_q: Current joint state (10 values: base3 + arm7)
            target_q: Target joint state (10 values: base3 + arm7)
            lock_base: If True, only plan arm motion (target_q[:3] ignored; base
                held at current_q[:3])

        Returns:
            (T, 10) numpy array, or None on failure
        """
        from curobo.types.robot import JointState
        from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig

        if not self._warmed_up:
            self.warmup()

        mg, q_active = self._select_mg(lock_base, current_q)
        joint_names = mg.robot_cfg.kinematics.cspace.joint_names
        target_active = np.asarray(target_q[3:10] if lock_base else target_q[:10],
                                   dtype=np.float32)

        current_state = JointState.from_position(
            torch.as_tensor(q_active, dtype=torch.float32,
                            device=self._device).unsqueeze(0),
            joint_names=joint_names,
        )

        goal_state = JointState.from_position(
            torch.as_tensor(target_active, dtype=torch.float32,
                            device=self._device).unsqueeze(0),
            joint_names=joint_names,
        )

        plan_config = MotionGenPlanConfig(
            max_attempts=10,
            enable_graph=True,
            enable_opt=True,
        )

        t0 = time.time()
        try:
            result = mg.plan_single_js(current_state, goal_state, plan_config)
            # Mirror plan_pose's recovery: if start-state collision, retry with
            # cleared world, then restore the real world so future calls see
            # the actual obstacles (plan_pose self-heals via _update_collision_for_target
            # at the top of each call; plan_joints has no such entry hook).
            if not result.success[0] and "START_STATE" in str(result.status):
                print(f"[curobo] Joint plan start-state collision, retrying with cleared world")
                from curobo.geom.types import WorldConfig as WC2, Cuboid as Cb2
                mg.update_world(WC2(cuboid=[
                    Cb2(name="ground", pose=[0,0,-1,1,0,0,0], dims=[0.01,0.01,0.01])
                ]))
                result = mg.plan_single_js(current_state, goal_state, plan_config)
                if self._world_cuboids:
                    robot_pos = (np.asarray(current_q[:2], dtype=np.float32)
                                 if len(current_q) >= 2 else None)
                    self.set_collision_world(self._world_cuboids, robot_pos=robot_pos)
        except Exception as e:
            dt = time.time() - t0
            print(f"[curobo] Joint plan ERROR ({dt:.2f}s): {e}")
            import traceback; traceback.print_exc()
            return None
        dt = time.time() - t0

        if result.success[0]:
            traj = result.get_interpolated_plan()
            positions = traj.position.cpu().numpy()
            positions = self._widen(positions, current_q, lock_base)
            print(f"[curobo] Joint plan OK ({dt:.2f}s, {'arm_only' if lock_base else 'whole_body'}): "
                  f"{positions.shape[0]} waypoints")
            return positions
        else:
            print(f"[curobo] Joint plan FAILED ({dt:.2f}s): {result.status}")
            return None

    def solve_ik(self, target_pos: np.ndarray, target_quat: np.ndarray,
                 current_q: Optional[np.ndarray] = None,
                 lock_base: bool = False,
                 num_seeds: int = 40,
                 return_closest: bool = True) -> Optional[np.ndarray]:
        """Solve IK for target EE pose.

        Args:
            target_pos: Target EE position [x, y, z]
            target_quat: Target EE orientation [w, x, y, z] (wxyz)
            current_q: Current joint state (10 values: base3 + arm7).
                Required when lock_base=True (used to pin base values).
                When return_closest=True, also used to pick solution closest to current.
            lock_base: If True, only solve for arm joints (base held at current_q[:3])
            num_seeds: Number of GPU-parallel IK seeds (mplib n_init_qpos equivalent)
            return_closest: If True and current_q provided, pick the returned solution
                with minimum joint-space distance to current_q. Otherwise return cuRobo's
                best (lowest cost) solution.

        Returns:
            (10,) numpy array of joint positions, or None on failure
        """
        from curobo.types.math import Pose

        if not self._warmed_up:
            self.warmup()

        if lock_base and current_q is None:
            raise ValueError("lock_base=True requires current_q to pin base values")

        # Same target-exclusion collision filter as plan_pose: exclude cuboids
        # within 0.5m of target so IK can find solutions that reach INTO fixtures
        # (grasping into drawers/counters). Without this, targets inside a fixture
        # AABB are unsolvable since EE would register as in-collision.
        if self._world_cuboids:
            robot_pos = current_q[:2] if current_q is not None and len(current_q) >= 2 else None
            self._update_collision_for_target(np.asarray(target_pos), robot_pos=robot_pos)

        # Select MotionGen based on mode (and update base locks if needed)
        if lock_base:
            mg, _ = self._select_mg(True, current_q)
        else:
            mg = self._motion_gen

        pos_np = np.asarray(target_pos, dtype=np.float32).reshape(1, 3)
        quat_np = np.asarray(target_quat, dtype=np.float32).reshape(1, 4)
        goal = Pose(
            position=torch.as_tensor(pos_np, device=self._device),
            quaternion=torch.as_tensor(quat_np, device=self._device),
        )

        # Ask cuRobo for multiple solutions so we can pick closest to current_q.
        # If return_closest is False or no current_q, 1 return_seed is enough.
        want_multi = bool(return_closest and current_q is not None)
        return_seeds = min(num_seeds, 8) if want_multi else 1

        t0 = time.time()
        try:
            result = mg.solve_ik(goal, num_seeds=num_seeds, return_seeds=return_seeds)
        except Exception as e:
            print(f"[curobo] IK ERROR ({time.time()-t0:.2f}s): {e}")
            return None
        dt = time.time() - t0

        if not bool(result.success.any()):
            print(f"[curobo] IK FAILED ({dt:.2f}s, num_seeds={num_seeds})")
            return None

        # result.solution shape: (B=1, return_seeds, dof). Pick best.
        sol = result.solution[0].cpu().numpy()          # (return_seeds, dof)
        succ = result.success[0].cpu().numpy().astype(bool)  # (return_seeds,)
        # Filter to successful solutions only
        sol_ok = sol[succ]
        if sol_ok.shape[0] == 0:
            return None

        if want_multi:
            # Pick closest to current_q in the active DOF space
            ref = current_q[3:10] if lock_base else current_q[:10]
            ref = np.asarray(ref, dtype=sol_ok.dtype)
            dists = np.linalg.norm(sol_ok - ref[None, :], axis=1)
            chosen = sol_ok[int(np.argmin(dists))]
        else:
            chosen = sol_ok[0]

        print(f"[curobo] IK OK ({dt:.2f}s, num_seeds={num_seeds}, "
              f"{int(succ.sum())}/{len(succ)} succeeded)")

        return self._widen(chosen, current_q if current_q is not None else np.zeros(10),
                           lock_base)


def build_collision_cuboids_from_fixtures(scene, fixtures_dict) -> list[dict]:
    """Extract kitchen fixture AABBs and convert to cuboid format.

    Reuses _compute_fixture_aabb from planning_utils to compute world-frame AABBs,
    then converts to center + half_size format for cuRobo.

    Args:
        scene: SAPIEN sub_scene
        fixtures_dict: dict of fixture name -> fixture object

    Returns:
        list of {"name": str, "center": [x,y,z], "half_size": [hx,hy,hz]}
    """
    from maniskill_tidyverse.planning_utils import _compute_fixture_aabb

    cuboids = []
    for fname, fix in fixtures_dict.items():
        # Skip floors and walls
        try:
            from robocasa.models.fixtures import Floor, Wall
            if isinstance(fix, (Floor, Wall)):
                continue
        except ImportError:
            pass

        if not hasattr(fix, 'pos'):
            continue

        bbox_min, bbox_max = _compute_fixture_aabb(scene, fname)

        # Fallback: use fixture pos+size
        if bbox_min is None and hasattr(fix, 'size') and fix.size is not None:
            pos = np.array(fix.pos)
            size = np.array(fix.size)
            if len(size) == 3 and np.all(size > 0.01):
                fix_half = size / 2.0
                bbox_min = pos - fix_half
                bbox_max = pos + fix_half

        if bbox_min is None:
            continue

        center = (bbox_min + bbox_max) / 2.0
        half_size = (bbox_max - bbox_min) / 2.0

        # Skip degenerate boxes
        if np.any(half_size < 0.005) or np.any(half_size > 5.0):
            continue

        cuboids.append({
            "name": f"fixture_{fname}",
            "center": center.tolist(),
            "half_size": half_size.tolist(),
        })

    print(f"[curobo] Extracted {len(cuboids)} fixture cuboids")
    return cuboids
