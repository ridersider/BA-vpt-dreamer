import os

import cv2
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter


class VideoLogger:
    """
    Records evaluation episodes as MP4 files and logs them (+ scalar metrics)
    to TensorBoard.

    Usage:
        video_logger = VideoLogger(log_dir="/workspace/logs", fps=20, record_every=10)
        # inside training loop:
        video_logger.log_metrics(update_num, metrics_dict)
        video_logger.maybe_record(update_num, actor, env)
    """

    def __init__(self, log_dir: str, fps: int = 20, record_every: int = 10, max_steps: int = 800):
        self.fps = fps
        self.record_every = record_every
        self.video_dir = os.path.join(log_dir, "videos")
        self.max_steps = max_steps
        os.makedirs(self.video_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))

    def log_metrics(self, update_num: int, metrics: dict):
        """Write scalar metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, global_step=update_num)
        self.writer.flush()

    def maybe_record(self, update_num: int, actor, env):
        """
        If update_num is a multiple of record_every, run one full evaluation
        episode (no exploration noise, same stochastic policy) and save the
        frames as MP4 + TensorBoard video.
        """
        if update_num % self.record_every != 0:
            return

        frames = self._collect_episode(actor, env, self.max_steps)
        if len(frames) == 0:
            return

        self._save_mp4(frames, update_num)
        self._log_tb_video(frames, update_num)

    # ── private helpers ────────────────────────────────────────────────────

    def _collect_episode(self, actor, env, max_steps: int = 400) -> list:
        """Run one episode (up to max_steps) and return (H, W, 3) uint8 RGB frames."""
        frames = []
        obs = env.reset()
        hidden = actor.get_initial_hidden_state()
        done = False

        for _ in range(max_steps):
            frames.append(obs["pov"].copy())
            env_action, _, _, _, hidden, _ = actor.step(obs, hidden, done)
            obs, _, done, _ = env.step(env_action)
            if done:
                break

        return frames

    def _save_mp4(self, frames: list, update_num: int):
        """Save frames as an MP4 file using OpenCV."""
        if not frames:
            return
        H, W, _ = frames[0].shape
        path = os.path.join(self.video_dir, f"update_{update_num:05d}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (W, H))
        for frame in frames:
            # OpenCV expects BGR
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[VideoLogger] Saved {path}  ({len(frames)} frames)")

    def _log_tb_video(self, frames: list, update_num: int):
        """
        Log video to TensorBoard.
        SummaryWriter.add_video expects shape (N, T, C, H, W) float32 in [0, 1].
        """
        if not frames:
            return
        # Stack to (T, H, W, C), normalise, reorder to (1, T, C, H, W)
        vid = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        vid = th.from_numpy(vid).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        self.writer.add_video("eval/episode", vid, global_step=update_num, fps=self.fps)
        self.writer.flush()

    def close(self):
        self.writer.close()
