#!/usr/bin/env python3
"""RoboCasa kitchen pick-and-place test.

Usage:
    python test_robocasa_pick_place.py --render human --seed 0
    python test_robocasa_pick_place.py --render rgb_array --seed 0
    python test_robocasa_pick_place.py --render rgb_array --seed 0 --max-tasks 3
    python test_robocasa_pick_place.py --render rgb_array --acm strict --viz-dir /tmp/viz
"""
import sys, os, signal, argparse, json, time, warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.WARNING)

import numpy as np
import torch, sapien, cv2
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs

from mplib import Pose as MPPose
from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld

from motion_utils import (
    ARM_HOME, GRIPPER_OPEN, GRIPPER_CLOSED, MASK_ARM_ONLY, MASK_WHOLE_BODY,
    get_robot_qpos, make_action, wait_until_stable, execute_trajectory,
    actuate_gripper,
)
from grasp_strategies import select_grasps, build_place_poses, DROP_HEIGHT
from planning_utils import sync_planner, build_kitchen_acm  # also applies monkey-patch
from video_utils import VideoWriter, CollisionLogger
from placement_utils import (
    CUBE_HALF, COLORS, spawn_cube, collect_placements,
)
from success_utils import compute_step_flags, format_flags
from viz_planning_world import save_planning_world


# ─── Constants ────────────────────────────────────────────────────────────────

LIFT_HEIGHT = 0.10
PLANNING_TIMEOUT = 15
IK_TIMEOUT = 8


# ─── Timeout handler ─────────────────────────────────────────────────────────

def _timeout_handler(signum, frame):
    raise TimeoutError("planning timeout")

signal.signal(signal.SIGALRM, _timeout_handler)


# ─── Pick-and-place attempt ──────────────────────────────────────────────────

def attempt_pick_place(task_idx, cube_pos, src_label, src_ftype,
                       dest_pos, dest_label, dest_ftype,
                       robot, planner, pw, step_fn, env, total,
                       timings, scene, agent, cube_actor,
                       src_fixture, dest_fixture,
                       viz_dir=None, step_label=None):
    """Pick from source, place at destination. Returns result dict."""
    original_pos = cube_pos.copy()
    phase_flags = []

    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    tag = f"[{task_idx+1}/{total}] {src_label}->{dest_label}"

    def _set_label(phase):
        if step_label is not None:
            step_label[0] = f"{tag} | {phase}"

    def _record_flags(phase_name, fixture=None):
        fix = fixture if fixture is not None else src_fixture
        flags = compute_step_flags(scene, agent, cube_actor, fix,
                                   original_pos, phase_name)
        phase_flags.append(flags)
        print(f"    Flags[{phase_name}]: {format_flags(flags)}")

    def _snap(stage_name, gname_s=""):
        if viz_dir:
            sync_planner(planner)
            slug = src_label.lower().replace(' ', '_').replace('/', '_')[:40]
            g_slug = gname_s.lower().replace('-', '_').replace(' ', '_')
            save_planning_world(
                pw, os.path.join(viz_dir, f"{task_idx}_{slug}_{g_slug}_{stage_name}"))

    def hold_open():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_OPEN, q[:3])

    def hold_closed():
        q = get_robot_qpos(robot)
        return make_action(q[3:10], GRIPPER_CLOSED, q[:3])

    # --- Try grasp strategies ---
    # Extract block yaw from its current pose so grasps align with faces
    from transforms3d.euler import quat2euler
    cube_q = cube_actor.pose.q
    if hasattr(cube_q, 'cpu'):
        cube_q = cube_q[0].cpu().numpy()
    else:
        cube_q = np.asarray(cube_q)
    _, _, obj_yaw = quat2euler(cube_q)
    ordered = select_grasps(cube_pos, arm_base, src_ftype, src_label,
                            obj_yaw=obj_yaw)

    grasped = False
    used_arm_only = False

    for gname, target_p, target_q in ordered:
        _set_label(f"IK ({gname})")
        print(f"\n  --- {tag} ({gname}) ---")

        grasp_pose = MPPose(p=target_p, q=target_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)

        # 1. Solve grasp IK
        q_grasp = None
        grasp_mask = None
        for mask_name, mask in [("arm-only", MASK_ARM_ONLY),
                                ("whole-body", MASK_WHOLE_BODY)]:
            t0 = time.time()
            signal.alarm(IK_TIMEOUT)
            try:
                status, solutions = planner.IK(
                    planner._transform_goal_to_wrt_base(grasp_pose),
                    cq, mask=mask, n_init_qpos=40, return_closest=True)
            except TimeoutError:
                timings['ik'] += time.time() - t0
                print(f"    IK ({mask_name}): TIMEOUT")
                continue
            finally:
                signal.alarm(0)
            dt = time.time() - t0
            timings['ik'] += dt
            if solutions is not None:
                q_grasp = solutions
                grasp_mask = mask
                print(f"    Grasp IK ({mask_name}): OK  [{dt:.2f}s]")
                break
            else:
                print(f"    Grasp IK ({mask_name}): no solution  [{dt:.2f}s]")

        if q_grasp is None:
            continue

        used_arm_only = bool(isinstance(grasp_mask, np.ndarray) and grasp_mask[0])
        motion_mask = MASK_ARM_ONLY if used_arm_only else MASK_WHOLE_BODY

        # 2. Plan approach to grasp pose
        _set_label(f"Approach ({gname})")
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_app = planner.plan_pose(grasp_pose, cq, mask=motion_mask,
                                       planning_time=5.0)
        except TimeoutError:
            r_app = {'status': 'TIMEOUT'}
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
        _record_flags('approach')
        _snap("1_approach", gname)

        # 3. Close gripper
        _set_label(f"Grasping ({gname})")
        t0 = time.time()
        actuate_gripper(step_fn, robot, GRIPPER_CLOSED)
        timings['gripper'] += time.time() - t0
        _record_flags('grasp')
        _snap("2_grasped", gname)

        grasped = True
        break  # got a grasp, proceed to place

    if not grasped:
        return {'outcome': 'unreachable', 'phases': phase_flags}

    # --- Lift ---
    _set_label("Lift")
    lift_pose = MPPose(p=target_p + np.array([0, 0, LIFT_HEIGHT]), q=target_q)
    sync_planner(planner)
    cq = get_robot_qpos(robot)
    t0 = time.time()
    signal.alarm(PLANNING_TIMEOUT)
    try:
        r_lift = planner.plan_pose(lift_pose, cq, mask=motion_mask,
                                    planning_time=5.0)
    except TimeoutError:
        r_lift = {'status': 'TIMEOUT'}
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
    else:
        print(f"    Lift: FAILED — {r_lift['status']}  [{dt:.2f}s]")
    _record_flags('lift')
    _snap("3_lifted")

    # --- Move to place pose (try multiple orientations) ---
    place_candidates = build_place_poses(dest_pos, arm_base)
    placed = False
    for pname, place_p, place_q in place_candidates:
        _set_label(f"Place ({pname})")
        place_pose = MPPose(p=place_p, q=place_q)
        sync_planner(planner)
        cq = get_robot_qpos(robot)
        t0 = time.time()
        signal.alarm(PLANNING_TIMEOUT)
        try:
            r_place = planner.plan_pose(place_pose, cq, mask=motion_mask,
                                         planning_time=5.0)
        except TimeoutError:
            r_place = {'status': 'TIMEOUT'}
        finally:
            signal.alarm(0)
        dt = time.time() - t0
        timings['planning'] += dt

        if r_place['status'] == 'Success':
            print(f"    Place ({pname}): OK ({r_place['position'].shape[0]} wp)  [{dt:.2f}s]")
            t0 = time.time()
            execute_trajectory(r_place['position'], step_fn, GRIPPER_CLOSED,
                               lock_base=used_arm_only, robot=robot)
            timings['exec'] += time.time() - t0
            placed = True
            break
        else:
            print(f"    Place ({pname}): FAILED — {r_place['status']}  [{dt:.2f}s]")
    if not placed:
        print(f"    Place: all {len(place_candidates)} orientations failed")

    # --- Open gripper ---
    _set_label("Release")
    t0 = time.time()
    actuate_gripper(step_fn, robot, GRIPPER_OPEN)
    for _ in range(30):
        step_fn(hold_open())
    timings['gripper'] += time.time() - t0
    _record_flags('release', fixture=dest_fixture)
    _snap("4_released")

    # --- Return home ---
    _set_label("Return home")
    sync_planner(planner)
    cq = get_robot_qpos(robot)
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
        print(f"    Return: OK ({r_home['position'].shape[0]} wp)  [{dt:.2f}s]")
        t0 = time.time()
        execute_trajectory(r_home['position'], step_fn, GRIPPER_OPEN,
                           robot=robot)
        timings['exec'] += time.time() - t0
    else:
        print(f"    Return: FAILED, teleporting  [{dt:.2f}s]")
        robot.set_qpos(torch.tensor(home_qpos, dtype=torch.float32).unsqueeze(0))

    t0 = time.time()
    wait_until_stable(step_fn,
                      make_action(ARM_HOME, GRIPPER_OPEN,
                                  get_robot_qpos(robot)[:3]),
                      robot, max_steps=100)
    timings['settle'] += time.time() - t0
    _record_flags('done', fixture=dest_fixture)

    # Determine outcome from flags
    done_flags = phase_flags[-1]
    grasp_flags = next((f for f in phase_flags if f['phase'] == 'grasp'), None)
    was_grasped = grasp_flags['is_grasped'] if grasp_flags else False
    at_dest = done_flags['obj_at_source']  # checked against dest_fixture
    gripper_clear = done_flags['gripper_far']

    if at_dest and gripper_clear:
        outcome = 'success'
    elif was_grasped and not at_dest:
        outcome = 'partial'
    elif not was_grasped:
        outcome = 'grasp_fail'
    else:
        outcome = 'partial'

    return {'outcome': outcome, 'phases': phase_flags}


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RoboCasa pick-and-place test")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human',
                        choices=['human', 'rgb_array'])
    parser.add_argument('--max-tasks', type=int, default=None,
                        help='Limit to N nearest tasks')
    parser.add_argument('--dest-label', type=str, default=None,
                        help='Substring match for fixed destination '
                             '(default: nearest placement)')
    parser.add_argument('--acm', default='strict',
                        choices=['relaxed', 'strict'])
    parser.add_argument('--viz-dir', type=str, default=None,
                        help='Save planning-world meshes (glb) at each stage')
    args = parser.parse_args()

    t_total = time.time()

    # --- Environment ---
    t0 = time.time()
    print("Creating RoboCasa env...")
    env = gym.make('RoboCasaKitchen-v1', num_envs=1,
                   robot_uids='tidyverse', control_mode='whole_body',
                   render_mode=args.render)
    env.reset(seed=args.seed)
    t_env = time.time() - t0
    print(f"  env setup: {t_env:.2f}s")

    robot = env.unwrapped.agent.robot
    agent = env.unwrapped.agent
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    # --- Camera: higher + tilted down ---
    from mani_skill.utils import sapien_utils as _su
    _rpos = robot.pose.p[0].cpu().numpy()
    _cam_eye = [_rpos[0], _rpos[1] - 3.5, 3.5]
    _cam_target = [_rpos[0], _rpos[1] + 1.0, 0.8]
    _cam_pose = _su.look_at(_cam_eye, _cam_target)
    _p = _cam_pose.raw_pose[0].cpu().numpy()
    _sapien_pose = sapien.Pose(p=_p[:3], q=_p[3:])
    for cam in env.unwrapped._human_render_cameras.values():
        cam.camera.set_local_pose(_sapien_pose)

    # --- Video ---
    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    if args.render == 'rgb_array':
        base = f'robocasa_pp_seed{args.seed}_acm{args.acm}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)

    # --- Collision logger ---
    collision_dir = os.path.join(os.path.dirname(__file__), 'collision_images')
    collision_logger = CollisionLogger(
        robot, scene, env, collision_dir, render_mode=args.render)

    step_label = ["idle"]

    def _burn_label(frame, text):
        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.35, h / 900)
        thick = max(1, int(h / 500))
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 14, th_ + 12), (0, 0, 0), -1)
        cv2.putText(frame, text, (7, th_ + 6), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def step_fn(action):
        env.step(action)
        if is_human:
            env.render()
        elif video_writer is not None:
            frame = env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            frame = frame.astype(np.uint8).copy()
            _burn_label(frame, step_label[0])
            video_writer.add_frame(frame)
        collision_logger.check(step_label[0])

    # --- Base initialization ---
    base_qpos = get_robot_qpos(robot)[:3].copy()
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot pose: {robot.pose.p[0].cpu().numpy()}, base qpos: {base_qpos}")

    # --- Collect placements and build tasks ---
    all_placements = collect_placements(fixtures)
    print(f"\nFound {len(all_placements)} placement locations")

    reachable = []
    for label, pos, ftype, fix_obj in all_placements:
        cube_pos = pos.copy()
        cube_pos[2] += CUBE_HALF + 0.002
        dist = np.linalg.norm(arm_base - cube_pos)
        reachable.append((label, cube_pos, ftype, fix_obj, dist))
    reachable.sort(key=lambda x: x[4])

    # Build task list: pick a single fixed destination, all blocks go there
    rng = np.random.RandomState(args.seed)
    dest_entry = None
    if args.dest_label:
        dest_entry = next((r for r in reachable
                           if args.dest_label.lower() in r[0].lower()), None)
        if dest_entry is None:
            print(f"WARNING: --dest-label '{args.dest_label}' not found, "
                  f"using nearest placement")
    if dest_entry is None:
        dest_entry = reachable[0]
    d_label, d_pos, d_ftype, d_fix, d_dist = dest_entry
    print(f"\nDestination: {d_label} ({d_ftype}) dist={d_dist:.2f}m")

    tasks = []
    for i, (s_label, s_pos, s_ftype, s_fix, s_dist) in enumerate(reachable):
        if s_label == d_label:
            continue
        tasks.append({
            'src_label': s_label, 'src_pos': s_pos, 'src_ftype': s_ftype,
            'src_fix': s_fix, 'src_dist': s_dist,
            'dst_label': d_label, 'dst_pos': d_pos, 'dst_ftype': d_ftype,
            'dst_fix': d_fix,
        })

    tasks.sort(key=lambda t: t['src_dist'])
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]

    print(f"\nWill attempt {len(tasks)} pick-place tasks:")
    for i, t in enumerate(tasks):
        print(f"  [{i:2d}] {t['src_label'][:30]:30s} ({t['src_ftype']:12s}) "
              f"-> {t['dst_label'][:30]:30s} ({t['dst_ftype']:12s}) "
              f"dist={t['src_dist']:.2f}m")

    # --- Spawn cubes ---
    for i, t in enumerate(tasks):
        color = COLORS[i % len(COLORS)]
        cube_name = f"obj_{i}_{t['src_label']}"
        try:
            actor = spawn_cube(scene, cube_name, t['src_pos'], color)
            t['cube_name'] = cube_name
            t['cube_actor'] = actor
        except Exception as e:
            print(f"  Spawn failed for {t['src_label']}: {e}")
            t['cube_actor'] = None
    tasks = [t for t in tasks if t.get('cube_actor') is not None]

    # --- Stabilize ---
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_qpos)
    step_label[0] = "Stabilizing"
    print("\nStabilizing robot...")
    t0 = time.time()
    wait_until_stable(step_fn, hold, robot)
    t_stabilize = time.time() - t0
    print(f"  stabilize: {t_stabilize:.2f}s")

    # --- Setup planner ---
    print("Setting up SapienPlanner...")
    t0 = time.time()
    signal.alarm(30)
    try:
        pw = SapienPlanningWorld(scene, [robot._objs[0]])
        eef = next(n for n in pw.get_planned_articulations()[0]
                   .get_pinocchio_model().get_link_names() if 'eef' in n)
        planner = SapienPlanner(pw, move_group=eef)
    except TimeoutError:
        print("FATAL: planner setup timed out")
        env.close()
        return
    finally:
        signal.alarm(0)
    t_planner = time.time() - t0
    print(f"  planner setup: {t_planner:.2f}s")

    cube_names = {t['cube_name'] for t in tasks}
    t0 = time.time()
    build_kitchen_acm(pw, planner, cube_names, mode=args.acm,
                      robot_pos=arm_base)
    t_acm = time.time() - t0
    print(f"  ACM build: {t_acm:.2f}s")

    if args.viz_dir:
        sync_planner(planner)
        save_planning_world(pw, os.path.join(args.viz_dir, "initial_home"))

    # --- Pick-place loop ---
    timings = {'ik': 0.0, 'planning': 0.0, 'exec': 0.0,
               'gripper': 0.0, 'settle': 0.0}
    results = {'success': 0, 'partial': 0, 'grasp_fail': 0,
               'unreachable': 0, 'error': 0}
    all_results = []

    for ci, t in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(tasks)}] {t['src_label']} ({t['src_ftype']}) "
              f"-> {t['dst_label']} ({t['dst_ftype']})  "
              f"dist={t['src_dist']:.2f}m")

        step_label[0] = f"task {ci+1}/{len(tasks)}"
        try:
            result = attempt_pick_place(
                ci, t['src_pos'], t['src_label'], t['src_ftype'],
                t['dst_pos'], t['dst_label'], t['dst_ftype'],
                robot, planner, pw, step_fn, env, len(tasks),
                timings, scene, agent, t['cube_actor'],
                t['src_fix'], t['dst_fix'],
                viz_dir=args.viz_dir, step_label=step_label)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            result = {'outcome': 'error', 'phases': []}
            try:
                cq = get_robot_qpos(robot)
                cq[3:10] = ARM_HOME
                cq[10:] = 0.0
                robot.set_qpos(torch.tensor(cq, dtype=torch.float32).unsqueeze(0))
            except Exception:
                pass

        outcome = result['outcome']
        results[outcome] = results.get(outcome, 0) + 1
        print(f"  => {outcome.upper()}")

        all_results.append({
            'index': ci,
            'src_label': t['src_label'], 'src_ftype': t['src_ftype'],
            'dst_label': t['dst_label'], 'dst_ftype': t['dst_ftype'],
            'distance': float(t['src_dist']),
            'outcome': outcome,
            'phases': result.get('phases', []),
        })

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}/{len(tasks)}")

    print(f"\nPer-task flag summary:")
    for r in all_results:
        print(f"[{r['index']+1}/{len(tasks)}] "
              f"{r['src_label']} -> {r['dst_label']} => {r['outcome'].upper()}")
        for pf in r['phases']:
            print(f"  {pf['phase']:12s}: {format_flags(pf)}")

    collision_logger.summary()

    # --- Timing ---
    t_total_elapsed = time.time() - t_total
    print(f"\n{'='*60}")
    print("TIMING BREAKDOWN:")
    print(f"  env setup:       {t_env:7.2f}s")
    print(f"  stabilize:       {t_stabilize:7.2f}s")
    print(f"  planner setup:   {t_planner:7.2f}s")
    print(f"  ACM build:       {t_acm:7.2f}s")
    print(f"  --- per-task ---")
    print(f"  IK solving:      {timings['ik']:7.2f}s")
    print(f"  path planning:   {timings['planning']:7.2f}s")
    print(f"  traj execution:  {timings['exec']:7.2f}s")
    print(f"  gripper:         {timings['gripper']:7.2f}s")
    print(f"  settle:          {timings['settle']:7.2f}s")
    task_total = sum(timings.values())
    accounted = t_env + t_stabilize + t_planner + t_acm + task_total
    print(f"  --- totals ---")
    print(f"  task phases:     {task_total:7.2f}s")
    print(f"  accounted:       {accounted:7.2f}s")
    print(f"  wall clock:      {t_total_elapsed:7.2f}s")

    # --- JSON output ---
    json_path = os.path.join(video_dir,
                             f'pp_results_seed{args.seed}_acm{args.acm}.json')
    with open(json_path, 'w') as f:
        json.dump({
            'seed': args.seed, 'acm_mode': args.acm,
            'totals': results, 'tasks': all_results,
            'timing': {
                'env': t_env, 'stabilize': t_stabilize,
                'planner': t_planner, 'acm': t_acm,
                **timings, 'wall_clock': t_total_elapsed,
            },
        }, f, indent=2, default=str)
    print(f"\nJSON: {json_path}")

    if is_human:
        print("\nDone! Close the window to exit.")
        while True:
            try:
                q = get_robot_qpos(robot)
                env.step(make_action(q[3:10], GRIPPER_OPEN, q[:3]))
                env.render()
            except Exception:
                break
    else:
        if video_writer:
            video_writer.close()
        env.close()


if __name__ == '__main__':
    main()
