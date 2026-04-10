import os

import cv2
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter


class VideoLogger:
    """
    Records training rollout frames as MP4 files and logs them (+ scalar metrics)
    to TensorBoard. Frames are collected during the normal training rollout —
    no separate evaluation episode, no environment reset.

    Usage:
        video_logger = VideoLogger(log_dir="/workspace/logs", fps=20, record_every=10)
        # inside training loop, before rollout:
        video_logger.start_rollout(update_num)
        # inside rollout step:
        video_logger.add_frame(obs["pov"])
        # after PPO update:
        video_logger.log_metrics(update_num, metrics_dict)
        video_logger.maybe_save_video(update_num)
    """

    def __init__(self, log_dir: str, fps: int = 20, record_every: int = 10):
        self.fps = fps
        self.record_every = record_every
        self.video_dir = os.path.join(log_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=os.path.join(log_dir, "tb"))
        self._frame_buffer: list = []
        self._recording = False

    def start_rollout(self, update_num: int):
        """Call before each rollout. Starts buffering frames if this update will be recorded."""
        self._recording = ((update_num + 1) % self.record_every == 0)
        self._frame_buffer.clear()

    def add_frame(self, frame: np.ndarray):
        """Add a raw obs['pov'] frame during rollout (only buffered when recording)."""
        if self._recording:
            self._frame_buffer.append(frame.copy())

    def log_metrics(self, update_num: int, metrics: dict):
        """Write scalar metrics to TensorBoard."""
        for key, value in metrics.items():
            self.writer.add_scalar(f"train/{key}", value, global_step=update_num)
        self.writer.flush()

    def maybe_save_video(self, update_num: int):
        """Save buffered rollout frames as MP4 + TensorBoard video if applicable."""
        if not self._recording or not self._frame_buffer:
            return
        self._save_mp4(self._frame_buffer, update_num)
        self._log_tb_video(self._frame_buffer, update_num)
        self._frame_buffer.clear()
        self._recording = False

    # ── private helpers ────────────────────────────────────────────────────

    def _save_mp4(self, frames: list, update_num: int):
        """Save frames as an MP4 file using OpenCV."""
        H, W, _ = frames[0].shape
        path = os.path.join(self.video_dir, f"update_{update_num:05d}.mp4")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, self.fps, (W, H))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        print(f"[VideoLogger] Saved {path}  ({len(frames)} frames)")

    def _log_tb_video(self, frames: list, update_num: int):
        """
        Log video to TensorBoard.
        SummaryWriter.add_video expects shape (N, T, C, H, W) float32 in [0, 1].
        """
        vid = np.stack(frames, axis=0).astype(np.float32) / 255.0  # (T, H, W, C)
        vid = th.from_numpy(vid).permute(0, 3, 1, 2).unsqueeze(0)  # (1, T, C, H, W)
        self.writer.add_video("train/rollout", vid, global_step=update_num, fps=self.fps)
        self.writer.flush()

    def close(self):
        self.writer.close()
