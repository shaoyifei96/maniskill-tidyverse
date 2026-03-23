"""Execution helpers: constants, action construction, trajectory execution, grasp attempt."""
import os
import signal
import time

import numpy as np
import torch

from mplib import Pose as MPPose

from perception import perceive_objects
from planning_utils import sync_planner
from viz_planning_world import save_planning_world


# ─── Constants ────────────────────────────────────────────────────────────────

ARM_HOME = np.array([0.0, -0.785, 0.0, -2.356, 0.0, 1.913, 0.785])
GRIPPER_OPEN = 0.0
GRIPPER_CLOSED = 0.81
PRE_GRASP_HEIGHT = 0.08
LIFT_HEIGHT = 0.15

MASK_ARM_ONLY  = np.array([True] * 3 + [False] * 7 + [True] * 6)
MASK_WHOLE_BODY = np.array([False] * 3 + [False] * 7 + [True] * 6)

PLANNING_TIMEOUT = 15
IK_TIMEOUT = 8


# ─── Timeout handler ─────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("planning timeout")

signal.signal(signal.SIGALRM, _timeout_handler)


# ─── Motion Helpers ──────────────────────────────────────────────────────────

def make_action(arm_qpos, gripper, base_cmd):
    act = np.concatenate([arm_qpos, [gripper], base_cmd])
    return torch.tensor(act, dtype=torch.float32).unsqueeze(0)


def get_robot_qpos(robot):
    return robot.get_qpos().cpu().numpy()[0]


def wait_until_stable(step_fn, hold, robot, max_steps=300,
                      vel_thresh=1e-3, window=10):
    stable_count = 0
    for si in range(max_steps):
        step_fn(hold)
        qvel = robot.get_qvel().cpu().numpy()[0]
        if np.max(np.abs(qvel)) < vel_thresh:
            stable_count += 1
            if stable_count >= window:
                return si + 1
        else:
            stable_count = 0
    return max_steps


def execute_trajectory(traj, step_fn, gripper, lock_base=False, robot=None,
                       settle_thresh=0.01, settle_steps=100):
    base_cmd = traj[0, 0:3] if lock_base else None
    for i in range(traj.shape[0]):
        b = base_cmd if lock_base else traj[i, 0:3]
        step_fn(make_action(traj[i, 3:10], gripper, b))

    final_arm = traj[-1, 3:10]
    final_base = base_cmd if lock_base else traj[-1, 0:3]
    final_act = make_action(final_arm, gripper, final_base)

    if robot is not None:
        for si in range(settle_steps):
            step_fn(final_act)
            qpos = get_robot_qpos(robot)
            arm_err = np.max(np.abs(qpos[3:10] - final_arm))
            base_err = np.max(np.abs(qpos[0:3] - final_base))
            if arm_err < settle_thresh and base_err < settle_thresh:
                break


def actuate_gripper(step_fn, robot, gripper_val, n_steps=30):
    qpos = get_robot_qpos(robot)
    arm, base = qpos[3:10], qpos[0:3]
    action = make_action(arm, gripper_val, base)
    for _ in range(n_steps):
        step_fn(action)


# ─── Single Grasp Attempt ────────────────────────────────────────────────────

def attempt_grasp(perception, strategies, robot, planner, pw, step_fn,
                  timings, idx, total, env=None, spawned_names=None,
                  drop_pos=None, viz_dir=None):
    """Execute grasp pipeline for a perceived object. Returns outcome string.

    Pipeline:
      1. Solve pre-grasp IK (PRE_GRASP_HEIGHT above target)
      2. Plan & execute path to pre-grasp
      3. Re-perceive with wrist camera -> refine grasp target
      4. Plan & execute approach to refined grasp pose
      5. Close gripper -> lift -> drop -> return home
    """
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()

    _snap_count = [0]
    def _snap(stage):
        if viz_dir:
            sync_planner(planner)
            name = f"obj{idx}_{perception.name[:30]}_{_snap_count[0]:02d}_{stage}"
            save_planning_world(pw, os.path.join(viz_dir, name))
            _snap_count[0] += 1

    tag_base = f"[{idx+1}/{total}] {perception.name}"

    for strategy_name, target_p, target_q in strategies:
        tag = f"{tag_base} ({strategy_name})"
        print(f"\n  --- {tag} ---")
        print(f"    Target pos: {target_p}")

        pre_pose = MPPose(p=target_p + [0, 0, PRE_GRASP_HEIGHT], q=target_q)

        sync_planner(planner)
        cq = get_robot_qpos(robot)

        # 1. Solve pre-grasp IK
        pre_base = planner._transform_goal_to_wrt_base(pre_pose)
        pregrasp_sols = None
        grasp_mask = None
        for mask_name, mask in [("arm-only", MASK_ARM_ONLY),
                                ("whole-body", MASK_WHOLE_BODY)]:
            t0 = time.time()
            signal.alarm(IK_TIMEOUT)
            try:
                status, solutions = planner.IK(
                    pre_base, cq, mask=mask, n_init_qpos=40,
                    return_closest=True)
            except TimeoutError:
                dt = time.time() - t0
                print(f"    Pre-grasp IK ({mask_name}): TIMEOUT  [{dt:.2f}s]")
                timings['ik'] += dt
                continue
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            timings['ik'] += dt
            if solutions is not None:
                pregrasp_sols = solutions
                grasp_mask = mask
                print(f"    Pre-grasp IK ({mask_name}): OK  [{dt:.2f}s]")
                break
            else:
                print(f"    Pre-grasp IK ({mask_name}): no solution  [{dt:.2f}s]")

        if pregrasp_sols is None:
            print(f"    Pre-grasp IK: FAILED for {strategy_name}")
            continue

        # 2. Plan path: current -> pre-grasp
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        _snap("pregrasp_plan")
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            result = planner.plan_qpos([pregrasp_sols], cq, planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Pre-grasp path: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if result['status'] != 'Success':
            print(f"    Pre-grasp path: FAILED — {result['status']}  [{dt:.2f}s]")
            continue
        print(f"    Pre-grasp path: OK ({result['position'].shape[0]} wp)  [{dt:.2f}s]")

        used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
        motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

        # Execute pre-grasp
        t0 = time.time()
        execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, robot=robot)
        timings['exec'] += time.time() - t0

        # Check base tracking — did the base reach where it should be?
        actual_qpos = get_robot_qpos(robot)
        planned_base = result['position'][-1, 0:3]
        base_error = np.linalg.norm(actual_qpos[0:3] - planned_base)
        print(f"    Base tracking: planned={planned_base}, "
              f"actual={actual_qpos[0:3]}, error={base_error:.4f}m")
        if base_error > 0.10:
            print(f"    Base tracking error {base_error:.3f}m > 0.10m — skipping grasp")
            # Return home before trying next strategy
            home_qpos = actual_qpos.copy()
            home_qpos[3:10] = ARM_HOME
            home_qpos[10:] = 0.0
            sync_planner(planner)
            signal.alarm(PLANNING_TIMEOUT)
            try:
                r_home = planner.plan_qpos([home_qpos], actual_qpos, planning_time=5.0)
            except TimeoutError:
                r_home = {'status': 'TIMEOUT'}
            finally:
                signal.alarm(0)
            if r_home['status'] == 'Success':
                execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN, robot=robot)
            else:
                robot.set_qpos(torch.tensor(home_qpos, dtype=torch.float32).unsqueeze(0))
            continue

        # 3. Re-perceive with wrist camera at pre-grasp position
        refined_target_p = target_p  # fallback
        if env is not None:
            print(f"    Wrist re-perception...")
            qpos_now = get_robot_qpos(robot)
            hold = make_action(qpos_now[3:10], GRIPPER_OPEN, qpos_now[:3])
            obs_wrist, _, _, _, _ = env.step(hold)

            wrist_perceptions = perceive_objects(
                obs_wrist, env.unwrapped, camera_name='wrist_camera',
                target_names=spawned_names)

            # Find the target object in wrist detections
            wrist_match = None
            for wp in wrist_perceptions:
                if wp.name == perception.name:
                    wrist_match = wp
                    break

            if wrist_match is not None:
                old_p = refined_target_p
                refined_target_p = wrist_match.center_3d
                shift = np.linalg.norm(refined_target_p - old_p)
                print(f"    Wrist detected {perception.name}: "
                      f"refined pos={refined_target_p} (shift={shift:.4f}m)")
            else:
                print(f"    Wrist: {perception.name} not detected "
                      f"({len(wrist_perceptions)} objects seen), using base estimate")

        # 4. Approach to refined grasp pose
        approach_pose = MPPose(p=refined_target_p, q=target_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        _snap("approach_plan")
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_app = planner.plan_pose(approach_pose, cq, mask=motion_mask,
                                       planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Approach: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            continue
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_app['status'] != 'Success':
            print(f"    Approach: FAILED — {r_app['status']}  [{dt:.2f}s]")
            continue
        print(f"    Approach: OK ({r_app['position'].shape[0]} wp)  [{dt:.2f}s]")

        t0 = time.time()
        execute_trajectory(r_app['position'], step_fn, GRIPPER_OPEN,
                           lock_base=used_arm_only, robot=robot)
        timings['exec'] += time.time() - t0

        # 5. Close gripper
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_CLOSED)
        qpos = get_robot_qpos(robot)
        hold_closed = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
        for _ in range(20):
            step_fn(hold_closed)
        timings['gripper'] += time.time() - t0

        # 6. Lift
        lift_pose = MPPose(p=refined_target_p + [0, 0, LIFT_HEIGHT], q=target_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        _snap("lift_plan")
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_lift = planner.plan_pose(lift_pose, cq, mask=motion_mask,
                                        planning_time=5.0)
        except TimeoutError:
            dt = time.time() - t0
            print(f"    Lift: TIMEOUT  [{dt:.2f}s]")
            timings['planning'] += dt
            actuate_gripper(step_fn, robot, GRIPPER_OPEN)
            return 'partial'
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_lift['status'] == 'Success':
            print(f"    Lift: OK ({r_lift['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_lift['position'], step_fn, GRIPPER_CLOSED,
                               lock_base=used_arm_only, robot=robot)
            timings['exec'] += time.time() - t0

            qpos = get_robot_qpos(robot)
            hold = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
            for _ in range(30):
                step_fn(hold)
        else:
            print(f"    Lift: FAILED — {r_lift['status']}  [{dt:.2f}s]")

        # 7. Move to sink and drop
        if drop_pos is not None:
            # Relax collision between grasped object and robot for transport
            acm = pw.get_allowed_collision_matrix()
            robot_link_names = planner.pinocchio_model.get_link_names()
            obj_pw_name = None
            for on in pw.get_object_names():
                if perception.name in on:
                    obj_pw_name = on
                    break
            if obj_pw_name:
                for rl in robot_link_names:
                    acm.set_entry(rl, obj_pw_name, True)
                print(f"    Relaxed collision: robot <-> {obj_pw_name}")

            # Plan to drop position above sink (top-down orientation)
            drop_q = np.array([0, 1, 0, 0], dtype=float)  # top-down
            drop_pose = MPPose(p=drop_pos, q=drop_q)
            sync_planner(planner)
            cq = get_robot_qpos(robot)
            _snap("sink_plan")
            t0 = time.time()
            signal.alarm(PLANNING_TIMEOUT)
            try:
                r_drop = planner.plan_pose(drop_pose, cq,
                                           mask=MASK_WHOLE_BODY,
                                           planning_time=5.0)
            except TimeoutError:
                dt = time.time() - t0
                print(f"    Move-to-sink: TIMEOUT  [{dt:.2f}s]")
                timings['planning'] += dt
                r_drop = {'status': 'TIMEOUT'}
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            timings['planning'] += dt

            if r_drop['status'] == 'Success':
                print(f"    Move-to-sink: OK ({r_drop['position'].shape[0]} wp)  [{dt:.2f}s]")
                t0 = time.time()
                execute_trajectory(r_drop['position'], step_fn, GRIPPER_CLOSED,
                                   robot=robot)
                timings['exec'] += time.time() - t0
            else:
                print(f"    Move-to-sink: FAILED — {r_drop['status']}  [{dt:.2f}s]")

        # Drop
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_OPEN)
        qpos = get_robot_qpos(robot)
        hold_open = make_action(qpos[3:10], GRIPPER_OPEN, qpos[:3])
        for _ in range(30):
            step_fn(hold_open)
        timings['gripper'] += time.time() - t0

        # 8. Return to home
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        _snap("return_plan")
        home_qpos = cq.copy()
        home_qpos[3:10] = ARM_HOME
        home_qpos[10:] = 0.0
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_home = planner.plan_qpos([home_qpos], cq, planning_time=5.0)
        except TimeoutError:
            r_home = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_home['status'] == 'Success':
            print(f"    Return: OK  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                               robot=robot)
            timings['exec'] += time.time() - t0
        else:
            print(f"    Return: FAILED, teleporting home  [{dt:.2f}s]")
            robot.set_qpos(torch.tensor(home_qpos, dtype=torch.float32).unsqueeze(0))

        t0 = time.time()
        wait_until_stable(step_fn,
                          make_action(ARM_HOME, GRIPPER_OPEN,
                                      get_robot_qpos(robot)[:3]),
                          robot, max_steps=100)
        timings['settle'] += time.time() - t0
        return 'success'

    return 'unreachable'
