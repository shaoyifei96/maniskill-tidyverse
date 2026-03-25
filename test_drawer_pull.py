#!/usr/bin/env python3
"""Perceive a drawer handle, approach with front grasp, close gripper, pull open."""
import sys, os, signal, numpy as np, torch, cv2, sapien
import maniskill_tidyverse.tidyverse_agent, mani_skill.envs, gymnasium as gym
import maniskill_tidyverse.planning_utils  # monkey-patch
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mplib import Pose as MPPose
from mani_skill.utils import common
from maniskill_tidyverse.perception import find_handle_targets, perceive_by_seg_id
from maniskill_tidyverse.grasp_strategies import front_grasp_from_normal
from maniskill_tidyverse.planning_utils import sync_planner
from maniskill_tidyverse.execution import (ARM_HOME, GRIPPER_OPEN, GRIPPER_CLOSED,
                       MASK_ARM_ONLY, MASK_WHOLE_BODY,
                       make_action, get_robot_qpos, wait_until_stable,
                       execute_trajectory, actuate_gripper)

signal.signal(signal.SIGALRM, lambda s, f: (_ for _ in ()).throw(TimeoutError()))

PULL_DISTANCE = 0.15
VIDEO_DIR = os.path.expanduser('~/tidyverse_videos')
os.makedirs(VIDEO_DIR, exist_ok=True)


def main():
    import time as _time
    _t0 = _time.perf_counter()
    env = gym.make('RoboCasaKitchen-v1', num_envs=1, robot_uids='tidyverse',
                   control_mode='whole_body', obs_mode='rgb+depth+segmentation',
                   render_mode='rgb_array', sensor_configs=dict(shader_pack='default'))
    obs, _ = env.reset(seed=0)
    print(f"[TIMER] env init + reset: {_time.perf_counter() - _t0:.1f}s")
    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']

    # Reposition render camera – raise the viewpoint
    from mani_skill.utils import sapien_utils as _su
    _rpos = robot.pose.p[0].cpu().numpy()
    _cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]
    _cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]
    _cam_pose = _su.look_at(_cam_eye, _cam_target)
    _p = _cam_pose.raw_pose[0].cpu().numpy()
    _sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
    for cam in env.unwrapped._human_render_cameras.values():
        cam.camera.set_local_pose(_sapien_pose)

    # Video writer
    run = 0
    while os.path.exists(os.path.join(VIDEO_DIR, f'drawer_pull_run{run}.mp4')):
        run += 1
    video_path = os.path.join(VIDEO_DIR, f'drawer_pull_run{run}.mp4')
    writer = None

    def step_fn(action):
        nonlocal writer
        obs_s, _, _, _, _ = env.step(action)
        frame = env.render()
        if isinstance(frame, torch.Tensor): frame = frame.cpu().numpy()
        if frame.ndim == 4: frame = frame[0]
        frame = frame.astype(np.uint8)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                     30, (w, h))
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        return obs_s

    hold = make_action(ARM_HOME, GRIPPER_OPEN, get_robot_qpos(robot)[:3])
    _t1 = _time.perf_counter()
    wait_until_stable(step_fn, hold, robot)
    print(f"[TIMER] wait_until_stable: {_time.perf_counter() - _t1:.1f}s")

    # Register drawer links
    _t1 = _time.perf_counter()
    handle_targets = find_handle_targets(fixtures, env.unwrapped)
    try: del env.unwrapped.__dict__['segmentation_id_map']
    except KeyError: pass
    seg_map = env.unwrapped.segmentation_id_map
    for ht in handle_targets:
        seg_map[int(ht['link'].per_scene_id)] = ht['link']

    obs = step_fn(hold)
    print(f"[TIMER] find_handle_targets + seg setup: {_time.perf_counter() - _t1:.1f}s")

    # Perceive target drawer
    _t1 = _time.perf_counter()
    ht = next(h for h in handle_targets if h['fixture_name'] == 'stack_2_right_group_3')
    sid = int(ht['link'].per_scene_id)
    perc = perceive_by_seg_id(obs, sid, camera_name='base_camera')
    if perc is None or perc.surface_normal is None:
        print("Can't see drawer!")
        env.close()
        return

    normal = perc.surface_normal
    handle_pos = perc.center_3d
    approach = -normal / np.linalg.norm(normal)
    pull_dir = normal / np.linalg.norm(normal)  # pull = away from surface
    yaw = np.arctan2(approach[1], approach[0])

    print(f"Handle: {handle_pos}")
    print(f"Normal: {normal}")
    print(f"Yaw: {np.degrees(yaw):.1f}°")

    print(f"[TIMER] perception: {_time.perf_counter() - _t1:.1f}s")

    # Grasp orientation: front + 90° finger rotation for horizontal bar
    q_grasp = front_grasp_from_normal(normal, rotate_fingers=True)

    # Setup planner
    _t1 = _time.perf_counter()
    pw = SapienPlanningWorld(scene, [robot._objs[0]])
    eef_name = next(n for n in pw.get_planned_articulations()[0]
                    .get_pinocchio_model().get_link_names() if 'eef' in n)
    planner = SapienPlanner(pw, move_group=eef_name)

    # Limit panda_joint1 to ±15°
    joint_names = planner.pinocchio_model.get_joint_names()
    j1_idx = next(i for i, n in enumerate(joint_names) if 'panda_joint1' in n)
    lim = planner.joint_limits
    lim[j1_idx] = [np.radians(-15), np.radians(15)]
    planner.joint_limits = lim

    # Relax all collisions
    acm = pw.get_allowed_collision_matrix()
    robot_links = planner.pinocchio_model.get_link_names()
    for obj_name in pw.get_object_names():
        if obj_name not in robot_links:
            for rl in robot_links:
                acm.set_entry(rl, obj_name, True)
    for art_name in pw.get_articulation_names():
        if 'tidyverse' in art_name.lower():
            continue
        art = pw.get_articulation(art_name)
        for link_name in art.get_pinocchio_model().get_link_names():
            for rl in robot_links:
                acm.set_entry(rl, link_name, True)

    # Read drawer state before
    drawer_art = ht['articulation']
    def read_drawer():
        for j in drawer_art.get_active_joints():
            if 'slide' in j.name.lower():
                q = float(drawer_art.get_qpos()[0, 0])
                maxd = getattr(ht['fixture'], 'max_displacement', 0.4)
                return q / maxd
        return -1

    print(f"[TIMER] planner setup + ACM: {_time.perf_counter() - _t1:.1f}s")

    print(f"\nDrawer BEFORE: {read_drawer():.3f}")

    # --- 1. Plan to grasp pose ---
    _t1 = _time.perf_counter()
    grasp_pose = MPPose(p=handle_pos, q=q_grasp)
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    grasp_base = planner._transform_goal_to_wrt_base(grasp_pose)

    sol = None
    for mask_name, mask, n_ik in [("arm-only", MASK_ARM_ONLY, 40),
                                   ("whole-body", MASK_WHOLE_BODY, 200)]:
        signal.alarm(8)
        try:
            _, s = planner.IK(grasp_base, cq, mask=mask,
                              n_init_qpos=n_ik, return_closest=True)
        except TimeoutError:
            s = None
        finally:
            signal.alarm(0)
        if s is not None:
            sol = s
            print(f"  Grasp IK ({mask_name}): OK")
            break
        print(f"  Grasp IK ({mask_name}): no solution")

    print(f"[TIMER] grasp IK: {_time.perf_counter() - _t1:.1f}s")

    if sol is None:
        print("IK failed!")
        env.close()
        return

    _t1 = _time.perf_counter()
    sync_planner(planner)
    signal.alarm(15)
    try:
        result = planner.plan_qpos([sol], get_robot_qpos(robot), planning_time=5.0)
    except TimeoutError:
        result = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)

    print(f"[TIMER] grasp path planning: {_time.perf_counter() - _t1:.1f}s")

    if result['status'] != 'Success':
        print(f"  Path planning failed: {result['status']}")
        env.close()
        return

    lock = bool(isinstance(sol, np.ndarray) and sol.shape[0] > 3 and
                np.allclose(sol[:3], get_robot_qpos(robot)[:3], atol=0.01))
    print(f"  Executing to grasp ({result['position'].shape[0]} wp)...")
    _t1 = _time.perf_counter()
    execute_trajectory(result['position'], step_fn, GRIPPER_OPEN,
                       robot=robot)
    print(f"[TIMER] execute grasp trajectory: {_time.perf_counter() - _t1:.1f}s")

    # Verify
    eef_link = next(l for l in robot.get_links() if 'eef' in l.get_name())
    eef_pos = eef_link.pose.p[0].cpu().numpy()
    print(f"  EEF: {eef_pos}  target: {handle_pos}  err: {np.linalg.norm(eef_pos - handle_pos):.4f}m")

    # --- 2. Close gripper ---
    _t1 = _time.perf_counter()
    print("  Closing gripper...")
    actuate_gripper(step_fn, robot, GRIPPER_CLOSED, n_steps=40)
    qpos = get_robot_qpos(robot)
    hold_closed = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
    for _ in range(20):
        step_fn(hold_closed)
    print(f"[TIMER] close gripper: {_time.perf_counter() - _t1:.1f}s")

    # --- 3. Pull ---
    _t1 = _time.perf_counter()
    pull_pos = handle_pos + pull_dir * PULL_DISTANCE
    pull_pose = MPPose(p=pull_pos, q=q_grasp)
    sync_planner(planner)
    cq = get_robot_qpos(robot)

    # Try plan_screw first (straight line), fallback to plan_pose
    signal.alarm(15)
    try:
        r_pull = planner.plan_screw(pull_pose, cq, time_step=0.05)
    except TimeoutError:
        r_pull = {'status': 'TIMEOUT'}
    finally:
        signal.alarm(0)

    if r_pull['status'] != 'Success':
        print(f"  Pull screw: {r_pull['status']}, trying plan_pose...")
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        signal.alarm(15)
        try:
            r_pull = planner.plan_pose(pull_pose, cq, mask=MASK_WHOLE_BODY,
                                        planning_time=5.0)
        except TimeoutError:
            r_pull = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)

    if r_pull['status'] != 'Success':
        # Last resort: blend toward home
        print(f"  Pull pose: {r_pull['status']}, trying qpos retract...")
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        retract = cq.copy()
        retract[3:10] = cq[3:10] * 0.6 + ARM_HOME * 0.4
        signal.alarm(15)
        try:
            r_pull = planner.plan_qpos([retract], cq, planning_time=5.0)
        except TimeoutError:
            r_pull = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)

    print(f"[TIMER] pull planning: {_time.perf_counter() - _t1:.1f}s")

    if r_pull['status'] == 'Success':
        print(f"  Pull: OK ({r_pull['position'].shape[0]} wp)")
        _t1 = _time.perf_counter()
        execute_trajectory(r_pull['position'], step_fn, GRIPPER_CLOSED, robot=robot)
        print(f"[TIMER] execute pull trajectory: {_time.perf_counter() - _t1:.1f}s")
    else:
        print(f"  Pull: FAILED — {r_pull['status']}")

    # Settle
    _t1 = _time.perf_counter()
    qpos = get_robot_qpos(robot)
    hold = make_action(qpos[3:10], GRIPPER_CLOSED, qpos[:3])
    for _ in range(30):
        step_fn(hold)

    print(f"\nDrawer AFTER: {read_drawer():.3f}")

    # Release
    actuate_gripper(step_fn, robot, GRIPPER_OPEN, n_steps=30)

    # A few more frames
    qpos = get_robot_qpos(robot)
    hold = make_action(qpos[3:10], GRIPPER_OPEN, qpos[:3])
    for _ in range(30):
        step_fn(hold)

    print(f"[TIMER] settle + release: {_time.perf_counter() - _t1:.1f}s")

    if writer:
        writer.release()
        print(f"\nVideo saved: {video_path}")

    print(f"[TIMER] TOTAL: {_time.perf_counter() - _t0:.1f}s")
    env.close()
    print("Done.")


if __name__ == '__main__':
    main()
