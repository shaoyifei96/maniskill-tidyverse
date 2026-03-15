"""Video recording and collision logging for simulation runs."""
import os
import numpy as np
import torch
import cv2


class VideoWriter:
    def __init__(self, path, fps=30, max_width=512):
        self.path = path
        self.fps = fps
        self.max_width = max_width
        self.writer = None
        self.frame_count = 0

    def add_frame(self, frame):
        if frame.ndim == 4:
            frame = frame[0]
        h, w = frame.shape[:2]
        if w > self.max_width:
            scale = self.max_width / w
            frame = cv2.resize(frame, (self.max_width, int(h * scale)))
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


class CollisionLogger:
    def __init__(self, robot, scene, env, img_dir, render_mode='human'):
        self.robot = robot
        self.scene = scene
        self.env = env
        self.img_dir = img_dir
        self.render_mode = render_mode
        import shutil
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)
        os.makedirs(img_dir, exist_ok=True)
        self.robot_entity_names = {l.get_name() for l in robot.get_links()}
        self.seen_pairs = set()
        self.collision_count = 0
        self.step_count = 0

    def check(self, step_label=""):
        self.step_count += 1
        try:
            contacts = self.scene.get_contacts()
        except Exception:
            return
        for contact in contacts:
            if not contact.points:
                continue
            impulse = np.sum([pt.impulse for pt in contact.points], axis=0)
            if np.linalg.norm(impulse) < 1e-4:
                continue
            b0, b1 = contact.bodies[0], contact.bodies[1]
            name0 = b0.entity.name if b0.entity else str(b0)
            name1 = b1.entity.name if b1.entity else str(b1)
            is_robot0 = name0 in self.robot_entity_names
            is_robot1 = name1 in self.robot_entity_names
            if not (is_robot0 or is_robot1):
                continue
            if is_robot0 and is_robot1:
                continue
            pair = frozenset((name0, name1))
            if pair not in self.seen_pairs:
                self.seen_pairs.add(pair)
                self.collision_count += 1
                robot_part = name0 if is_robot0 else name1
                other_part = name1 if is_robot0 else name0
                sep = min(pt.separation for pt in contact.points)
                imp_mag = np.linalg.norm(impulse)
                print(f"  COLLISION #{self.collision_count} step={self.step_count}: "
                      f"{robot_part} <-> {other_part}  "
                      f"impulse={imp_mag:.4f}  sep={sep:.4f}  "
                      f"[{step_label}]")
                self._save_image(robot_part, other_part)

    def _save_image(self, robot_part, other_part):
        try:
            frame = self.env.render()
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            if frame.ndim == 4:
                frame = frame[0]
            img = frame.astype(np.uint8)
            text = (f"COLLISION #{self.collision_count} step={self.step_count}: "
                    f"{robot_part} <-> {other_part}")
            h = img.shape[0]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.4, h / 800)
            thick = max(1, int(h / 400))
            (tw, th_), _ = cv2.getTextSize(text, font, scale, thick)
            cv2.rectangle(img, (5, 5), (tw + 15, th_ + 15), (0, 0, 200), -1)
            cv2.putText(img, text, (10, th_ + 10), font, scale,
                        (255, 255, 255), thick, cv2.LINE_AA)
            safe_name = (f"collision_{self.collision_count:03d}"
                         f"_step{self.step_count:05d}"
                         f"_{robot_part}_vs_{other_part}.png").replace('/', '_')
            cv2.imwrite(os.path.join(self.img_dir, safe_name),
                        cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        except Exception:
            pass

    def summary(self):
        print(f"\nCollision summary: {self.collision_count} unique collision pairs "
              f"detected over {self.step_count} steps")
        for pair in sorted(self.seen_pairs, key=lambda p: sorted(p)):
            names = sorted(pair)
            print(f"  - {names[0]} <-> {names[1]}")
