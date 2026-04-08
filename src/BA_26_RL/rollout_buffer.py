from typing import Generator

import numpy as np
import torch as th


class VPTRolloutBuffer:
    """
    Stores one rollout of (obs, action, reward, done, value, log_prob) tuples
    and computes GAE advantages + lambda-returns.
    """

    def __init__(self, n_steps: int, image_shape: tuple, gamma: float, gae_lambda: float, device):
        self.n_steps = n_steps
        self.image_shape = image_shape  # (H, W, C)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        self.pos = 0

        H, W, C = image_shape
        self.obs_imgs = np.zeros((n_steps, H, W, C), dtype=np.uint8)
        self.buttons = np.zeros((n_steps, 1), dtype=np.int64)
        self.cameras = np.zeros((n_steps, 1), dtype=np.int64)
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)
        self.log_probs = np.zeros(n_steps, dtype=np.float32)
        self.advantages = np.zeros(n_steps, dtype=np.float32)
        self.returns = np.zeros(n_steps, dtype=np.float32)

    def reset(self):
        self.pos = 0

    def add(self, obs_img: np.ndarray, agent_action: dict, reward: float,
            done: bool, value: float, log_prob: float):
        """Store one transition. obs_img must be (H, W, C) uint8."""
        t = self.pos
        self.obs_imgs[t] = obs_img
        self.buttons[t] = agent_action["buttons"].cpu().numpy()
        self.cameras[t] = agent_action["camera"].cpu().numpy()
        self.rewards[t] = reward
        self.dones[t] = float(done)
        self.values[t] = value
        self.log_probs[t] = log_prob
        self.pos += 1

    def finalize(self, last_value: float, last_done: bool):
        """
        Compute GAE advantages and lambda-returns.
        Must be called after the rollout is complete (pos == n_steps).
        """
        last_gae_lam = 0.0
        for t in reversed(range(self.n_steps)):
            if t == self.n_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]
            delta = (
                self.rewards[t]
                + self.gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae_lam = (
                delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            )
            self.advantages[t] = last_gae_lam
        self.returns = self.advantages + self.values

    def get(self, batch_size: int) -> Generator[dict, None, None]:
        """Yield shuffled minibatches as dicts of tensors."""
        indices = np.random.permutation(self.n_steps)
        start = 0
        while start < self.n_steps:
            batch_idx = indices[start: start + batch_size]
            yield {
                "obs_imgs": self.obs_imgs[batch_idx],       # (B, H, W, C) uint8
                "buttons": th.from_numpy(self.buttons[batch_idx]).to(self.device),
                "cameras": th.from_numpy(self.cameras[batch_idx]).to(self.device),
                "old_log_probs": th.from_numpy(self.log_probs[batch_idx]).to(self.device),
                "old_values": th.from_numpy(self.values[batch_idx]).to(self.device),
                "advantages": th.from_numpy(self.advantages[batch_idx]).to(self.device),
                "returns": th.from_numpy(self.returns[batch_idx]).to(self.device),
            }
            start += batch_size
