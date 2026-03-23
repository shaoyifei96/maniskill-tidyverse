#!/usr/bin/env python3
"""Perception-based grasp pipeline: use depth + segmentation cameras to detect
objects, fit ellipses, choose grasp strategy, and execute with mplib planner.

Unlike test_robocasa_grasp.py which uses known object positions (spawns cubes),
this script uses the robot's cameras to perceive objects and infer grasps.

Usage:
    # GUI — watch the robot perceive and grasp
    python test_perception_grasp.py --render human --seed 0

    # Headless — save video
    python test_perception_grasp.py --render rgb_array --seed 0

    # Use a specific task env
    python test_perception_grasp.py --task RoboCasaKitchen-v1 --render human
"""
import sys, os, signal, argparse, time
import numpy as np
import torch, sapien, cv2
import gymnasium as gym

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tidyverse_agent   # noqa: F401 — registers 'tidyverse'
import mani_skill.envs    # noqa: F401 — registers envs
try:
    import robocasa_tasks  # noqa: registers RoboCasa single-stage tasks
except ImportError:
    pass

# Apply monkey-patch by importing planning_utils
import planning_utils  # noqa: F401 — patches SapienPlanningWorld

from mplib.sapien_utils import SapienPlanner, SapienPlanningWorld
from mani_skill.utils import common

from perception import (
    perceive_objects, classify_fixture_context, save_perception_debug,
    PerceptionResult, deproject_pixels_to_world,
)
from grasp_strategies import choose_grasp_strategy
from planning_utils import (
    add_fixture_boxes_to_planner, build_kitchen_acm, sync_planner,
    compute_articulation_aabb,
)
from execution import (
    ARM_HOME, GRIPPER_OPEN, GRIPPER_CLOSED,
    PRE_GRASP_HEIGHT, LIFT_HEIGHT,
    MASK_ARM_ONLY, MASK_WHOLE_BODY,
    PLANNING_TIMEOUT, IK_TIMEOUT,
    make_action, get_robot_qpos, wait_until_stable,
    execute_trajectory, actuate_gripper, attempt_grasp,
)
from viz_planning_world import save_planning_world


# ─── Object Spawning (optional — for testing without task env objects) ────────

def spawn_test_objects(scene, fixtures):
    """Spawn small objects on various surfaces for testing perception."""
    from test_robocasa_grasp import collect_placements, spawn_cube, COLORS, CUBE_HALF

    all_placements = collect_placements(fixtures)
    spawned = []
    for i, (label, pos, ftype) in enumerate(all_placements):
        color = COLORS[i % len(COLORS)]
        cube_pos = pos.copy()
        cube_pos[2] += CUBE_HALF + 0.002
        cube_name = f"test_obj_{i}_{label}"
        try:
            builder = scene.create_actor_builder()
            hs = np.array([CUBE_HALF] * 3)
            builder.add_box_collision(half_size=hs)
            builder.add_box_visual(
                half_size=hs,
                material=sapien.render.RenderMaterial(
                    base_color=color),
            )
            actor = builder.build(name=cube_name)
            actor.set_pose(sapien.Pose(p=cube_pos))
            spawned.append((cube_name, cube_pos, label, ftype, actor))
        except Exception as e:
            print(f"  Spawn failed for {label}: {e}")

    return spawned


# ─── Video Writer ─────────────────────────────────────────────────────────────

class VideoWriter:
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frame_count = 0

    def add_frame(self, frame):
        if frame.ndim == 4:
            frame = frame[0]
        h, w = frame.shape[:2]
        if self.writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))
        self.writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.frame_count += 1

    def close(self):
        if self.writer:
            self.writer.release()
            print(f"Video saved: {self.path} ({self.frame_count} frames)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Perception-based grasp pipeline")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--render', default='human', choices=['human', 'rgb_array'])
    parser.add_argument('--task', default='RoboCasaKitchen-v1',
                        help='Task env ID')
    parser.add_argument('--camera', default='base_camera',
                        help='Camera to use for perception (base_camera or wrist_camera)')
    parser.add_argument('--max-objects', type=int, default=None,
                        help='Max objects to attempt grasping')
    parser.add_argument('--spawn-test-objects', action='store_true',
                        help='Spawn test cubes on surfaces (if task has no objects)')
    parser.add_argument('--acm', default='strict',
                        choices=['relaxed', 'strict'],
                        help='ACM mode: relaxed=ignore all fixtures, '
                             'strict=only relax distant fixtures (default: strict)')
    parser.add_argument('--viz-dir', default=None,
                        help='Save planning world obstacle map snapshots')
    parser.add_argument('--debug-dir', default=None,
                        help='Save perception debug images to this directory')
    args = parser.parse_args()

    t_total = time.time()

    # --- Create environment with perception ---
    print(f"Creating env: {args.task} with rgb+depth+segmentation...")
    t0 = time.time()
    env = gym.make(
        args.task,
        num_envs=1,
        robot_uids='tidyverse',
        control_mode='whole_body',
        obs_mode='rgb+depth+segmentation',
        render_mode=args.render,
        sensor_configs=dict(shader_pack="default"),
    )
    obs, info = env.reset(seed=args.seed)
    t_env = time.time() - t0
    print(f"  env setup: {t_env:.2f}s")

    robot = env.unwrapped.agent.robot
    scene = env.unwrapped.scene.sub_scenes[0]
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']
    is_human = (args.render == 'human')

    # Reposition render camera
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
    video_dir = os.path.expanduser('~/tidyverse_videos')
    os.makedirs(video_dir, exist_ok=True)
    video_writer = None
    cam_video_writers = {}
    if args.render == 'rgb_array':
        base = f'perception_grasp_seed{args.seed}_acm{args.acm}'
        run = 0
        while os.path.exists(os.path.join(video_dir, f'{base}_run{run}.mp4')):
            run += 1
        video_path = os.path.join(video_dir, f'{base}_run{run}.mp4')
        video_writer = VideoWriter(video_path, fps=30)
        # Per-camera video writers
        for cam_name in ['base_camera', 'wrist_camera']:
            cam_path = os.path.join(video_dir, f'{base}_run{run}_{cam_name}.mp4')
            cam_video_writers[cam_name] = VideoWriter(cam_path, fps=30)

    step_label = ["idle"]

    def _burn_label(frame, text):
        h = frame.shape[0]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, h / 600)
        thick = max(1, int(h / 300))
        (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
        cv2.rectangle(frame, (0, 0), (tw + 20, th_ + 16), (0, 0, 0), -1)
        cv2.putText(frame, text, (10, th_ + 8), font, scale,
                    (255, 255, 255), thick, cv2.LINE_AA)

    def step_fn(action):
        obs_step, _, _, _, _ = env.step(action)
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
            # Save per-camera RGB frames
            for cam_name, cw in cam_video_writers.items():
                if cam_name in obs_step.get("sensor_data", {}):
                    cam_rgb = common.to_numpy(obs_step["sensor_data"][cam_name]["rgb"][0])
                    cam_frame = cam_rgb.astype(np.uint8).copy()
                    _burn_label(cam_frame, f"{cam_name}: {step_label[0]}")
                    cw.add_frame(cam_frame)

    robot_pos = robot.pose.p[0].cpu().numpy()
    arm_base = next(l for l in robot.get_links()
                    if l.get_name() == 'panda_link0').pose.p[0].cpu().numpy()
    print(f"Robot at {robot_pos}, arm base at {arm_base}")

    # --- Optionally spawn test objects ---
    spawned_names = None
    if args.spawn_test_objects:
        print("\nSpawning test objects on surfaces...")
        spawned = spawn_test_objects(scene, fixtures)
        spawned_names = {s[0] for s in spawned}
        print(f"  Spawned {len(spawned)} objects: {spawned_names}")
        # Step to let objects settle and update render
        hold = make_action(ARM_HOME, GRIPPER_OPEN, get_robot_qpos(robot)[:3])
        for _ in range(60):
            step_fn(hold)
        # Force segmentation ID map refresh and register spawned actors
        try:
            del env.unwrapped.__dict__['segmentation_id_map']
        except KeyError:
            pass
        # Manually inject spawned actors into segmentation_id_map
        seg_map = env.unwrapped.segmentation_id_map
        for name, pos, label, ftype, actor in spawned:
            sid = actor.per_scene_id
            seg_map[sid] = actor
            print(f"    Registered {name} with seg_id={sid}")

    # --- Stabilize ---
    base_cmd = get_robot_qpos(robot)[:3].copy()
    hold = make_action(ARM_HOME, GRIPPER_OPEN, base_cmd)
    step_label[0] = "Stabilizing"
    print("\nStabilizing robot...")
    t0 = time.time()
    wait_until_stable(step_fn, hold, robot)
    print(f"  stabilize: {time.time() - t0:.2f}s")

    # --- Get fresh observation for perception ---
    print("\nCapturing perception data...")
    step_label[0] = "Perceiving"
    obs, _, _, _, _ = env.step(hold)

    # List available cameras
    available_cameras = list(obs['sensor_data'].keys())
    print(f"  Available cameras: {available_cameras}")

    if not available_cameras:
        print("ERROR: No cameras available!")
        env.close()
        return

    # --- Run perception on ALL cameras, merge results ---
    t0 = time.time()
    perceptions = []
    seen_names = set()
    debug_dir = args.debug_dir or os.path.join(
        os.path.dirname(__file__), 'perception_debug')

    registered_ids = set(env.unwrapped.segmentation_id_map.keys()) if spawned_names else set()
    for cam_name in available_cameras:
        # Debug: dump what seg IDs are in the image
        _seg = common.to_numpy(obs["sensor_data"][cam_name]["segmentation"][0])[..., 0]
        _unique = set(np.unique(_seg).tolist())
        _overlap = _unique & registered_ids
        print(f"\n  [{cam_name}] seg IDs in image: {sorted(_unique)}")
        if spawned_names:
            print(f"    registered IDs: {sorted(registered_ids)}")
            print(f"    overlap: {sorted(_overlap)}")

        cam_perceptions = perceive_objects(obs, env, camera_name=cam_name,
                                           target_names=spawned_names)
        print(f"  [{cam_name}] detected {len(cam_perceptions)} objects")
        # Save per-camera debug images
        cam_debug_dir = os.path.join(debug_dir, cam_name)
        save_perception_debug(obs, env, cam_perceptions, cam_name, cam_debug_dir)

        # Merge, dedup by name (keep the one with more pixels)
        for p in cam_perceptions:
            p.fixture_context = classify_fixture_context(p.center_3d, fixtures)
            if p.name not in seen_names:
                seen_names.add(p.name)
                perceptions.append(p)
            else:
                for j, existing in enumerate(perceptions):
                    if existing.name == p.name and p.mask_pixels > existing.mask_pixels:
                        perceptions[j] = p
                        break

    t_percept = time.time() - t0
    print(f"\n  Perceived {len(perceptions)} unique objects in {t_percept:.2f}s:")

    # Sort by distance to arm base (nearest first)
    perceptions.sort(key=lambda p: np.linalg.norm(p.center_3d - arm_base))

    if args.max_objects is not None:
        perceptions = perceptions[:args.max_objects]

    for i, p in enumerate(perceptions):
        dist = np.linalg.norm(p.center_3d - arm_base)
        sz = p.size_3d
        print(f"  [{i:2d}] {p.name:40s} dist={dist:.2f}m  "
              f"size=[{sz[0]:.3f},{sz[1]:.3f},{sz[2]:.3f}]  "
              f"aspect={p.aspect_ratio:.1f}  ctx={p.fixture_context}")

    if not perceptions:
        print("\nNo objects detected — nothing to grasp.")
        env.close()
        return

    # --- Setup planner ---
    print("\nSetting up SapienPlanner...")
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

    # Add fixture AABB boxes to planning world (mplib only, no physics)
    # Skip sink — it's the drop target, we don't want collision there
    skip_set = set()
    for fname, fix in fixtures.items():
        if 'sink' in fname.lower():
            skip_set.add(fname)
    print(f"\n  Adding fixture box approximations to planner (skipping: {skip_set})...")
    fixture_box_names = add_fixture_boxes_to_planner(pw, scene, fixtures,
                                                      skip_fixtures=skip_set)
    print(f"  Added {len(fixture_box_names)} fixture boxes to planning world")

    # --- Find sink for placement ---
    from mani_skill.utils.scene_builder.robocasa.fixtures.sink import Sink
    sink_pos = None
    for fname, fix in fixtures.items():
        if isinstance(fix, Sink):
            sink_pos = np.array(fix.pos)
            print(f"\n  Sink found: {fname} at {sink_pos}")
            break
    if sink_pos is None:
        for fname, fix in fixtures.items():
            if 'sink' in fname.lower() and hasattr(fix, 'pos'):
                sink_pos = np.array(fix.pos)
                print(f"\n  Sink (by name): {fname} at {sink_pos}")
                break
    if sink_pos is not None:
        drop_pos = sink_pos.copy()
        drop_pos[2] += 0.2  # 0.2m above sink
        print(f"  Drop target: {drop_pos}")
    else:
        drop_pos = None
        print("\n  WARNING: No sink found — will drop in place")

    # ACM — keep collision checking for perceived objects
    # Use robot start + all grasp targets + sink as reference points
    target_names = {p.name for p in perceptions}
    target_positions = [p.center_3d for p in perceptions]
    if drop_pos is not None:
        target_positions.append(drop_pos)
    t0 = time.time()
    build_kitchen_acm(pw, planner, target_names, mode=args.acm,
                      robot_pos=arm_base, target_positions=target_positions)
    t_acm = time.time() - t0
    print(f"  ACM build: {t_acm:.2f}s")

    # --- Save obstacle map snapshots ---
    if args.viz_dir:
        os.makedirs(args.viz_dir, exist_ok=True)
        sync_planner(planner)
        save_planning_world(pw, os.path.join(args.viz_dir, "planning_world_all"))
        print(f"  Full planning world saved to {args.viz_dir}/planning_world_all.glb")

    # --- Grasp loop ---
    timings = {'ik': 0.0, 'planning': 0.0, 'exec': 0.0,
               'gripper': 0.0, 'settle': 0.0}
    results = {'success': 0, 'partial': 0, 'unreachable': 0, 'error': 0}

    for ci, perception in enumerate(perceptions):
        dist = np.linalg.norm(perception.center_3d - arm_base)
        print(f"\n{'='*60}")
        print(f"[{ci+1}/{len(perceptions)}] {perception.name} "
              f"({perception.fixture_context})  dist={dist:.2f}m")
        print(f"  center={perception.center_3d}  "
              f"size={perception.size_3d}  aspect={perception.aspect_ratio:.1f}")

        step_label[0] = f"grasp {ci+1}/{len(perceptions)} {perception.name}"

        # Choose grasp strategies based on perception
        strategies = choose_grasp_strategy(perception, arm_base)
        strategy_names = [s[0] for s in strategies]
        print(f"  Strategy order: {strategy_names}")

        try:
            outcome = attempt_grasp(perception, strategies, robot, planner, pw,
                                    step_fn, timings, ci, len(perceptions),
                                    env=env, spawned_names=spawned_names,
                                    drop_pos=drop_pos, viz_dir=args.viz_dir)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            outcome = 'error'
            try:
                cq = get_robot_qpos(robot)
                cq[3:10] = ARM_HOME
                cq[10:] = 0.0
                robot.set_qpos(torch.tensor(cq, dtype=torch.float32).unsqueeze(0))
            except Exception:
                pass

        results[outcome] = results.get(outcome, 0) + 1
        print(f"  => {outcome.upper()}")

    # --- Summary ---
    print(f"\n{'='*60}")
    print("RESULTS:")
    for k, v in results.items():
        print(f"  {k:12s}: {v}/{len(perceptions)}")

    t_total_elapsed = time.time() - t_total
    print(f"\nTIMING:")
    print(f"  env setup:     {t_env:7.2f}s")
    print(f"  perception:    {t_percept:7.2f}s")
    print(f"  planner setup: {t_planner:7.2f}s")
    print(f"  ACM build:     {t_acm:7.2f}s")
    print(f"  IK:            {timings['ik']:7.2f}s")
    print(f"  planning:      {timings['planning']:7.2f}s")
    print(f"  execution:     {timings['exec']:7.2f}s")
    print(f"  gripper:       {timings['gripper']:7.2f}s")
    print(f"  settle:        {timings['settle']:7.2f}s")
    print(f"  TOTAL:         {t_total_elapsed:7.2f}s")

    if video_writer:
        video_writer.close()
    for cw in cam_video_writers.values():
        cw.close()

    env.close()
    print("\nDone.")


if __name__ == '__main__':
    main()
