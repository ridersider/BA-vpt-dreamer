from typing import List

import gym
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler

LOG_KEYS = [
    "oak_log",     "birch_log",    "spruce_log",
    "jungle_log",  "acacia_log",   "dark_oak_log",
    "oak_wood",    "birch_wood",   "spruce_wood",
    "jungle_wood", "acacia_wood",  "dark_oak_wood",
]

REWARD_PER_LOG = 1.0
REWARD_PER_AXE = 5.0


class SurvivalEnv(HumanSurvival):
    """
    Standard Minecraft survival environment with normal world generation.
    No superflat world, no pre-placed trees — just vanilla survival.
    """

    def __init__(self, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "MineRLSurvival-v0"
        super().__init__(**kwargs)

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=6000),
            handlers.SpawningInitialCondition(allow_spawning=True),
            handlers.DefaultWorldGenerator(force_reset=True, generator_options='{"seed": "42"}'),
        ]


class SurvivalRewardEnv(gym.Wrapper):
    """
    Reward wrapper for SurvivalEnv:
      +1.0 per newly collected wood log (any type)
      +5.0 per newly crafted wooden axe
    """

    def __init__(self, env):
        super().__init__(env)
        self._prev_logs = 0
        self._prev_axes = 0

    def _count_logs(self, obs) -> int:
        inv = obs["inventory"]
        return sum(int(inv[k]) for k in LOG_KEYS)

    def _count_axes(self, obs) -> int:
        return int(obs["inventory"].get("wooden_axe", 0))

    def reset(self):
        obs = self.env.reset()
        self._prev_logs = self._count_logs(obs)
        self._prev_axes = self._count_axes(obs)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        logs = self._count_logs(obs)
        axes = self._count_axes(obs)

        reward = (
            REWARD_PER_LOG * max(0, logs - self._prev_logs)
            + REWARD_PER_AXE * max(0, axes - self._prev_axes)
        )

        self._prev_logs = logs
        self._prev_axes = axes

        return obs, reward, done, info
