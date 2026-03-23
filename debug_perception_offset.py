"""Compare ground-truth actor poses vs camera back-projected perception positions."""
import os, sys, numpy as np, torch, gymnasium as gym

# Register agent
sys.path.insert(0, os.path.dirname(__file__))
import tidyverse_agent  # noqa: F401
from mani_skill.utils import common

from perception import perceive_objects
from execution import make_action, ARM_HOME, GRIPPER_OPEN, wait_until_stable, get_robot_qpos
from test_perception_grasp import spawn_test_objects


def main():
    print("Creating env...")
    env = gym.make(
        'RoboCasaKitchen-v1',
        num_envs=1,
        robot_uids='tidyverse',
        control_mode='whole_body',
        obs_mode='rgb+depth+segmentation',
        render_mode='rgb_array',
        sensor_configs=dict(shader_pack="default"),
    )
    obs, _ = env.reset(seed=0)
    scene = env.unwrapped.scene.sub_scenes[0]
    robot = env.unwrapped.agent.robot

    # Get fixtures dict (same as main script)
    fixtures = env.unwrapped.scene_builder.scene_data[0]['fixtures']

    spawned = spawn_test_objects(scene, fixtures)
    print(f"Spawned {len(spawned)} objects")

    # Step to settle
    hold = make_action(ARM_HOME, GRIPPER_OPEN, get_robot_qpos(robot)[:3])
    for _ in range(60):
        env.step(hold)

    # Register seg IDs
    try:
        del env.unwrapped.__dict__['segmentation_id_map']
    except KeyError:
        pass
    seg_map = env.unwrapped.segmentation_id_map
    spawned_actors = {}
    for name, pos, label, ftype, actor in spawned:
        sid = actor.per_scene_id
        seg_map[sid] = actor
        spawned_actors[name] = actor

    # Stabilize
    wait_until_stable(lambda a: env.step(a), hold, robot)

    # Fresh obs
    obs, _, _, _, _ = env.step(hold)

    # Get cam2world for each camera + inspect depth format
    for cam_name in ['base_camera', 'wrist_camera']:
        sensor_params = obs["sensor_param"][cam_name]
        cam2world = common.to_numpy(sensor_params["cam2world_gl"][0])
        depth_raw = common.to_numpy(obs["sensor_data"][cam_name]["depth"][0])
        print(f"\n{'='*60}")
        print(f"Camera: {cam_name}")
        print(f"  cam2world_gl:\n{cam2world}")
        print(f"  cam position (world): {cam2world[:3, 3]}")
        print(f"  depth dtype: {depth_raw.dtype}, shape: {depth_raw.shape}")
        print(f"  depth range: [{depth_raw.min():.4f}, {depth_raw.max():.4f}]")
        print(f"  depth sample (center pixel): {depth_raw[depth_raw.shape[0]//2, depth_raw.shape[1]//2]}")

    # Run perception + ellipse-center comparison
    import cv2
    from perception import deproject_pixels_to_world

    spawned_names = {s[0] for s in spawned}
    for cam_name in ['base_camera', 'wrist_camera']:
        sensor_data = obs["sensor_data"][cam_name]
        depth = common.to_numpy(sensor_data["depth"][0])[..., 0]
        seg = common.to_numpy(sensor_data["segmentation"][0])[..., 0]
        sensor_params = obs["sensor_param"][cam_name]
        intrinsic = common.to_numpy(sensor_params["intrinsic_cv"][0])
        cam2world = common.to_numpy(sensor_params["cam2world_gl"][0])

        perceptions = perceive_objects(obs, env, camera_name=cam_name,
                                       target_names=spawned_names)
        if not perceptions:
            print(f"\n[{cam_name}] No detections")
            continue

        print(f"\n{'='*60}")
        print(f"[{cam_name}] Comparison: bbox_mid vs ellipse_center_depth")
        print(f"{'name':>50s}  {'GT':>21s}  {'bbox_mid':>21s} {'err':>6s}  {'ellipse':>21s} {'err':>6s}")

        bbox_errors, ell_errors = [], []
        for p in perceptions:
            actor = spawned_actors.get(p.name)
            if actor is None:
                continue
            gt_pos = np.array(actor.pose.p)

            # Method 1: bbox midpoint (current)
            bbox_mid = p.center_3d
            bbox_diff = np.linalg.norm(bbox_mid - gt_pos)
            bbox_errors.append(bbox_mid - gt_pos)

            # Method 2: ellipse center + depth at that pixel
            mask = seg == p.seg_id
            ys, xs = np.where(mask)
            pixels = np.stack([xs, ys], axis=-1)
            ell_pos = None
            ell_diff = float('nan')
            if len(pixels) >= 5:
                try:
                    ellipse = cv2.fitEllipse(pixels.astype(np.float32))
                    ecx, ecy = int(round(ellipse[0][0])), int(round(ellipse[0][1]))
                    # Clamp to image bounds
                    ecx = max(0, min(ecx, depth.shape[1] - 1))
                    ecy = max(0, min(ecy, depth.shape[0] - 1))
                    # Get depth at ellipse center
                    ed = depth[ecy, ecx]
                    if ed > 0:
                        ell_pos = deproject_pixels_to_world(
                            np.array([[ecx, ecy]]), depth, intrinsic, cam2world)[0]
                        ell_diff = np.linalg.norm(ell_pos - gt_pos)
                        ell_errors.append(ell_pos - gt_pos)
                except cv2.error:
                    pass

            gt_str = f"[{gt_pos[0]:6.3f},{gt_pos[1]:7.3f},{gt_pos[2]:6.3f}]"
            bm_str = f"[{bbox_mid[0]:6.3f},{bbox_mid[1]:7.3f},{bbox_mid[2]:6.3f}]"
            if ell_pos is not None:
                el_str = f"[{ell_pos[0]:6.3f},{ell_pos[1]:7.3f},{ell_pos[2]:6.3f}]"
            else:
                el_str = "       N/A            "
            print(f"{p.name:>50s}  {gt_str}  {bm_str} {bbox_diff:5.3f}m  {el_str} {ell_diff:5.3f}m")

        if bbox_errors:
            be = np.array(bbox_errors)
            print(f"\n  bbox_mid  — mean L2: {np.mean(np.linalg.norm(be, axis=1)):.4f}m  "
                  f"bias: dx={np.mean(be[:,0]):+.4f} dy={np.mean(be[:,1]):+.4f} dz={np.mean(be[:,2]):+.4f}")
        if ell_errors:
            ee = np.array(ell_errors)
            print(f"  ellipse   — mean L2: {np.mean(np.linalg.norm(ee, axis=1)):.4f}m  "
                  f"bias: dx={np.mean(ee[:,0]):+.4f} dy={np.mean(ee[:,1]):+.4f} dz={np.mean(ee[:,2]):+.4f}")

    env.close()


if __name__ == '__main__':
    main()
