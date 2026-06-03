"""
Navigation environment: agent moves to a random XZ target within target_radius.

Reward per step:  (prev_xz_dist - curr_xz_dist) * reward_scale
                  Positive when approaching, negative when retreating, zero when still.
Success bonus:    +success_bonus when within success_radius — episode ends.

Target is sampled each reset: uniform random point inside a circle of
target_radius blocks around the agent's actual spawn position (Y ignored).

When a survival_spec (SurvivalEnv) is provided (navigation mode only), a red
wool tower is placed at the target position via DrawingDecorator so the agent
can see the goal from spawn. The tower spans y=1..255. Because the target
depends on spawn (only known after reset), the tower is placed using the spawn
cached from the previous episode. The first episode has no tower; from episode 2
onwards the tower is always at the correct target. With a fixed world seed the
spawn position is deterministic, so the cached value matches perfectly.
"""

import json
import math
import os
import random
from typing import List, Optional, Tuple

import gym

_TOWER_Y_MIN = 1
_TOWER_Y_MAX = 255


def _tower_xml(x: int, z: int) -> str:
    return (
        f'<DrawCuboid x1="{x}" y1="{_TOWER_Y_MIN}" z1="{z}" '
        f'x2="{x}" y2="{_TOWER_Y_MAX}" z2="{z}" type="wool" colour="RED"/>'
    )


class NavigationRewardEnv(gym.Wrapper):
    """
    Reward wrapper for XZ-plane navigation to a randomly sampled target.

    Continuous reward = distance-delta shaping (potential-based, Ng et al. 1999):
        r_t = (d_{t-1} - d_t) * reward_scale

    Sparse success bonus applied and episode terminated when dist <= success_radius.

    If log_dir is given, each episode's trajectory is written to
        <log_dir>/nav_episodes/episode_<N>.json

    survival_spec: optional SurvivalEnv spec reference (navigation mode only).
        When provided, a red wool tower is generated at the target each episode.
    """

    def __init__(
        self,
        env: gym.Env,
        target_radius: float = 300.0,
        success_radius: float = 5.0,
        reward_scale: float = 1.0,
        success_bonus: float = 10.0,
        log_dir: Optional[str] = None,
        survival_spec=None,
    ):
        super().__init__(env)
        self._target_radius = target_radius
        self._success_radius = success_radius
        self._reward_scale = reward_scale
        self._success_bonus = success_bonus
        self._survival_spec = survival_spec

        self._target_x: float = 0.0
        self._target_z: float = 0.0
        self._spawn_x: float = 0.0
        self._spawn_z: float = 0.0
        self._prev_dist: float = 0.0
        self._episode_return: float = 0.0
        self._episode_length: int = 0
        self._episode_count: int = 0
        self._trajectory: List[Tuple[float, float]] = []

        # Spawn cached from the previous episode.
        # None = no episode has completed yet → skip tower on first reset.
        self._cached_spawn_x: Optional[float] = None
        self._cached_spawn_z: Optional[float] = None

        self._log_dir: Optional[str] = os.path.join(log_dir, "nav_episodes") if log_dir else None
        if self._log_dir:
            os.makedirs(self._log_dir, exist_ok=True)

    def _xz(self, obs) -> Tuple[float, float]:
        return float(obs["location_stats"]["xpos"]), float(obs["location_stats"]["zpos"])

    def _xz_dist(self, x: float, z: float) -> float:
        return math.sqrt((x - self._target_x) ** 2 + (z - self._target_z) ** 2)

    def _sample_target_from(self, spawn_x: float, spawn_z: float) -> Tuple[float, float]:
        angle = random.uniform(0, 2 * math.pi)
        r = math.sqrt(random.uniform(0, 1)) * self._target_radius  # uniform in disk
        return spawn_x + r * math.cos(angle), spawn_z + r * math.sin(angle)

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
        # Place the red wool tower before env.reset() so it appears in the mission XML.
        # We pre-sample the target using the cached spawn from the previous episode.
        # (Spawn is deterministic with a fixed world seed, so this is exact from ep 2 on.)
        if self._survival_spec is not None:
            if self._cached_spawn_x is not None:
                pre_target_x, pre_target_z = self._sample_target_from(
                    self._cached_spawn_x, self._cached_spawn_z
                )
                self._survival_spec.set_drawing_decorator(
                    _tower_xml(round(pre_target_x), round(pre_target_z))
                )
            else:
                # First episode: spawn not yet known, skip tower.
                self._survival_spec.set_drawing_decorator(None)
                pre_target_x = pre_target_z = None

        obs = self.env.reset()
        self._spawn_x, self._spawn_z = self._xz(obs)
        self._cached_spawn_x = self._spawn_x
        self._cached_spawn_z = self._spawn_z

        if self._survival_spec is not None and pre_target_x is not None:
            self._target_x, self._target_z = pre_target_x, pre_target_z
        else:
            self._target_x, self._target_z = self._sample_target_from(self._spawn_x, self._spawn_z)

        self._prev_dist = self._xz_dist(self._spawn_x, self._spawn_z)
        self._episode_return = 0.0
        self._episode_length = 0
        self._trajectory = [[self._spawn_x, self._spawn_z]]
        tower_note = " (no tower — first episode)" if (
            self._survival_spec is not None and pre_target_x is None
        ) else ""
        print(f"[NavigationEnv] spawn=({self._spawn_x:.1f}, {self._spawn_z:.1f})  "
              f"target=({self._target_x:.1f}, {self._target_z:.1f})  "
              f"dist={self._prev_dist:.1f}m{tower_note}")
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
