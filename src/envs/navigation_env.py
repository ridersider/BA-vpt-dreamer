"""
Navigation environment: agent moves to a random XZ target within target_radius.

Reward per step:  (prev_xz_dist - curr_xz_dist) * reward_scale
                  Positive when approaching, negative when retreating, zero when still.
Success bonus:    +success_bonus when within success_radius — episode ends.

Target is sampled each reset: uniform random point inside a circle of
target_radius blocks around the agent's actual spawn position (Y ignored).
"""

import json
import math
import os
import random
from typing import List, Optional, Tuple

import gym


class NavigationRewardEnv(gym.Wrapper):
    """
    Reward wrapper for XZ-plane navigation to a randomly sampled target.

    Continuous reward = distance-delta shaping (potential-based, Ng et al. 1999):
        r_t = (d_{t-1} - d_t) * reward_scale

    Sparse success bonus applied and episode terminated when dist <= success_radius.

    If log_dir is given, each episode's trajectory is written to
        <log_dir>/nav_episodes/episode_<N>.json
    """

    def __init__(
        self,
        env: gym.Env,
        target_radius: float = 300.0,
        success_radius: float = 5.0,
        reward_scale: float = 1.0,
        success_bonus: float = 10.0,
        log_dir: Optional[str] = None,
    ):
        super().__init__(env)
        self._target_radius = target_radius
        self._success_radius = success_radius
        self._reward_scale = reward_scale
        self._success_bonus = success_bonus

        self._target_x: float = 0.0
        self._target_z: float = 0.0
        self._spawn_x: float = 0.0
        self._spawn_z: float = 0.0
        self._prev_dist: float = 0.0
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_count: int = 0
        self._trajectory: List[Tuple[float, float]] = []

        self._log_dir: Optional[str] = os.path.join(log_dir, "nav_episodes") if log_dir else None
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

    def _xz(self, obs) -> Tuple[float, float]:
        return float(obs["location_stats"]["xpos"]), float(obs["location_stats"]["zpos"])

    def _xz_dist(self, x: float, z: float) -> float:
        return math.sqrt((x - self._target_x) ** 2 + (z - self._target_z) ** 2)

    def _sample_target(self) -> Tuple[float, float]:
        angle = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(0, 1)) * self._target_radius  # uniform in disk
        return self._spawn_x + r * math.cos(angle), self._spawn_z + r * math.sin(angle)

    def _write_episode_json(self, success: bool) -> None:
        if not self._log_dir:
            return
        data = {
            "episode": self._episode_count,
            "spawn":   [self._spawn_x, self._spawn_z],
            "target":  [self._target_x, self._target_z],
            "success": success,
            "length":  self._episode_length,
            "return":  self._episode_return,
            "trajectory": self._trajectory,
        }
        path = os.path.join(self._log_dir, f"episode_{self._episode_count:05d}.json")
        with open(path, "w") as f:
            json.dump(data, f)

    def reset(self):
        obs = self.env.reset()
        self._spawn_x, self._spawn_z = self._xz(obs)
        self._target_x, self._target_z = self._sample_target()
        self._prev_dist = self._xz_dist(self._spawn_x, self._spawn_z)
        self._episode_return = 0.0
        self._episode_length = 0
        self._trajectory = [[self._spawn_x, self._spawn_z]]
        print(f"[NavigationEnv] spawn=({self._spawn_x:.1f}, {self._spawn_z:.1f})  "
              f"target=({self._target_x:.1f}, {self._target_z:.1f})  "
              f"dist={self._prev_dist:.1f}m")
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        x, z = self._xz(obs)
        curr_dist = self._xz_dist(x, z)
        reward = (self._prev_dist - curr_dist) * self._reward_scale

        success = curr_dist <= self._success_radius
        if success:
            reward += self._success_bonus
            done = True

        self._episode_return += reward
        self._episode_length += 1
        self._prev_dist = curr_dist
        self._trajectory.append([x, z])

        info.update(
            nav_dist=curr_dist,
            nav_target=(self._target_x, self._target_z),
            nav_success=success,
        )
        if done:
            self._write_episode_json(success)
            self._episode_count += 1
            info.update(
                nav_episode_return=self._episode_return,
                nav_episode_length=self._episode_length,
            )

        return obs, reward, done, info
