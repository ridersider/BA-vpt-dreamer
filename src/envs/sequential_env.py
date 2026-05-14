"""
Sequential curriculum environment.

Tasks are defined in config.TASKS and come in two types:

  obtain  goal_key="reward_based"
          Task completes when accumulated reward reaches task.max_reward
          (i.e. the agent collected the full capped quantity of every item).

  craft   goal_key=<item_id>
          Task completes when the goal item is newly acquired.
          If action_reward_key is set, a one-time reward is given each time
          that action is triggered (e.g. "inventory" for the 2x2 crafting grid).

Info keys exposed per step:
    task_idx           int   index of the currently active task
    task_name          str   display name of the active task
    task_complete      bool  True on the step the task is completed
    all_tasks_complete bool  True when the last task is completed
    player_died        bool  True when the episode ended due to player death
"""

from __future__ import annotations

import gym
import numpy as np

from config import TASKS, TaskSpec  # noqa: F401 (TaskSpec used in type hints)


class SequentialRewardEnv(gym.Wrapper):

    def __init__(self, env: gym.Env, start_task_idx: int = 0, repeat: bool = False):
        super().__init__(env)
        self._start_task_idx: int = start_task_idx
        self._task_idx: int = start_task_idx
        self._repeat: bool = repeat
        self._prev_inventory: dict = {}  # item_id → last seen count (updated every step)
        self._rewarded: dict = {}        # item_id → units already rewarded (never decreases)
        self._task_reward: float = 0.0   # accumulated reward in current task
        self._action_open: bool = False  # toggle state for action_reward_key
        self._goal_baseline: int = 0     # inventory count of goal item at task start

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def task_idx(self) -> int:
        return self._task_idx

    @property
    def task(self) -> TaskSpec:
        return TASKS[self._task_idx]

    # ── Inventory helpers ─────────────────────────────────────────────────────

    def _read_counts(self, obs) -> dict:
        inv = obs["inventory"]
        return {
            item.item_id: (
                sum(int(inv.get(k, 0)) for k in item.variants)
                if item.variants
                else int(inv.get(item.item_id, 0))
            )
            for item in self.task.reward_items
        }

    def _read_goal_count(self, obs) -> int:
        """Read the current inventory count of the task's goal item."""
        return int(obs["inventory"].get(self.task.goal_key, 0))

    def _snapshot(self, obs) -> None:
        """Reset per-task accumulators. Items already in inventory count as new
        so the agent gets immediate reward for resources carried over from previous tasks."""
        counts = self._read_counts(obs)
        self._prev_inventory = {item_id: 0 for item_id in counts}
        self._rewarded = {item_id: 0 for item_id in counts}
        self._task_reward = 0.0
        self._action_open = False
        self._goal_baseline = (
            self._read_goal_count(obs)
            if self.task.task_type == "craft"
            else 0
        )

    # ── gym.Wrapper interface ─────────────────────────────────────────────────

    def seed(self, seed: int = 42):
        return self.env.seed(seed)

    def reset(self):
        obs = self.env.reset()
        self._task_idx = self._start_task_idx
        self._snapshot(obs)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)

        if done:
            self._task_idx = self._start_task_idx
            self._snapshot(obs)
            info.update(
                task_idx=self._task_idx,
                task_name=self.task.display_name,
                task_complete=False,
                all_tasks_complete=False,
                player_died=True,
            )
            return obs, 0.0, done, info

        reward = 0.0
        task_complete = False

        # ── Drop penalty (Q button) ────────────────────────────────────────────
        #if action.get("drop", 0):
        #    reward -= 0.5

        # ── Action reward (sequential: given every step while UI is open) ──────
        if self.task.action_reward_key and action.get(self.task.action_reward_key, 0):
            self._action_open = not self._action_open  # key press toggles open/closed

        if self._action_open:
            reward += self.task.action_reward

        # ── Item rewards ──────────────────────────────────────────────────────
        current = self._read_counts(obs)
        for item in self.task.reward_items:
            prev = self._prev_inventory.get(item.item_id, 0)
            curr = current[item.item_id]
            new_items = max(0, curr - prev)

            if new_items > 0:
                already = self._rewarded.get(item.item_id, 0)
                if item.max_qty == float("inf"):
                    reward += item.reward_per_item * new_items
                    self._rewarded[item.item_id] = already + new_items
                else:
                    rewardable = max(0, min(already + new_items, item.max_qty) - already)
                    reward += item.reward_per_item * rewardable
                    self._rewarded[item.item_id] = min(already + new_items, item.max_qty)

                if self.task.task_type == "craft" and item.item_id == self.task.goal_key:
                    task_complete = True

            self._prev_inventory[item.item_id] = curr  # always track true current count

        self._task_reward += reward

        # ── Task completion ───────────────────────────────────────────────────
        if self.task.task_type == "obtain":
            if self._task_reward >= self.task.max_reward:
                task_complete = True
        elif self.task.goal_key not in self.task.item_reward_keys:
            # Goal not tracked via item rewards → check inventory directly
            if self._read_goal_count(obs) > self._goal_baseline:
                task_complete = True

        all_tasks_complete = False
        if task_complete:
            if self._repeat:
                self._snapshot(obs)
            elif self._task_idx < len(TASKS) - 1:
                self._task_idx += 1
                self._snapshot(obs)
            else:
                all_tasks_complete = True
                done = True

        info.update(
            task_idx=self._task_idx,
            task_name=self.task.display_name,
            task_complete=task_complete,
            all_tasks_complete=all_tasks_complete,
        )
        if task_complete:
            info["inventory_snapshot"] = {
                k: int(v) for k, v in obs["inventory"].items() if int(v) > 0
            }
        return obs, reward, done, info
