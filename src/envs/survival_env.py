from typing import Dict, List, Optional, Tuple

import gym
import minerl.herobraine.hero.handlers as handlers
from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from minerl.herobraine.hero.handler import Handler

from config import MASTER_REWARD_TABLE, get_reward_item, RewardItem


class SurvivalEnv(HumanSurvival):
    """Vanilla Minecraft survival with a fixed world seed and optional start inventory."""

    def __init__(self, start_inventory: Tuple[Tuple[str, int], ...] = (), **kwargs):
        self._start_inventory = start_inventory
        self._drawing_decorator_xml: Optional[str] = None
        if "name" not in kwargs:
            kwargs["name"] = "MineRLSurvival-v0"
        super().__init__(**kwargs)

    def create_server_initial_conditions(self) -> List[Handler]:
        return [
            handlers.TimeInitialCondition(allow_passage_of_time=True, start_time=6000),
            handlers.SpawningInitialCondition(allow_spawning=True),
            handlers.DefaultWorldGenerator(force_reset=True, generator_options='{"seed": "42"}'),
        ]

    def create_server_decorators(self) -> List[Handler]:
        if self._drawing_decorator_xml:
            return [handlers.DrawingDecorator(self._drawing_decorator_xml)]
        return []

    def set_drawing_decorator(self, xml: Optional[str]) -> None:
        """Set the DrawingDecorator XML used in the next episode's mission XML."""
        self._drawing_decorator_xml = xml

    def create_agent_start(self) -> List[Handler]:
        base = super().create_agent_start()
        if not self._start_inventory:
            return base
        inv = {
            slot: {"type": item_id, "quantity": qty}
            for slot, (item_id, qty) in enumerate(self._start_inventory)
        }
        return base + [handlers.InventoryAgentStart(inv)]

    def set_start_inventory(self, items: Tuple[Tuple[str, int], ...]) -> None:
        """Update the start inventory used in the next reset's mission XML."""
        self._start_inventory = items


class SurvivalRewardEnv(gym.Wrapper):
    """
    Reward wrapper driven by config.MASTER_REWARD_TABLE.

    item_ids        Optional list of item_ids to reward. When None, all items in
                    MASTER_REWARD_TABLE are rewarded (full VPT paper reward table).
    keep_inventory  When True, the agent's full inventory at the moment of death is
                    restored at the next reset by updating the underlying SurvivalEnv
                    spec before the new mission is sent to Malmo.
    survival_spec   The SurvivalEnv instance (spec, not the made env) — required when
                    keep_inventory=True so that set_start_inventory() can be called
                    before each reset.

    Item log
    --------
    Every step where an item count increases is appended to self.item_log as:
        {"step": int, "item_id": str, "delta": int, "total": int}
    All inventory keys are tracked (not just reward items), so the log can be
    used to plot the full collection history after the run.
    """

    def __init__(
        self,
        env: gym.Env,
        item_ids: Optional[List[str]] = None,
        keep_inventory: bool = True,
        survival_spec: Optional[SurvivalEnv] = None,
    ):
        super().__init__(env)
        self._reward_items: List[RewardItem] = (
            [get_reward_item(i) for i in item_ids]
            if item_ids is not None
            else list(MASTER_REWARD_TABLE)
        )
        self._prev_inventory: dict = {}
        self._rewarded: dict = {}

        self._keep_inventory = keep_inventory
        self._survival_spec = survival_spec
        self._item_log: List[dict] = []
        self._step_count: int = 0
        self._prev_full_inventory: Dict[str, int] = {}
        self._pending_inventory: Optional[Tuple] = None  # set on death, applied on next reset

    @property
    def item_log(self) -> List[dict]:
        """List of {"step", "item_id", "delta", "total"} entries — one per pickup event."""
        return self._item_log

    def _read_counts(self, obs) -> dict:
        inv = obs["inventory"]
        return {
            item.item_id: (
                sum(int(inv.get(k, 0)) for k in item.variants)
                if item.variants
                else int(inv.get(item.item_id, 0))
            )
            for item in self._reward_items
        }

    def _read_full_inventory(self, obs) -> Dict[str, int]:
        return {k: int(v) for k, v in obs["inventory"].items() if int(v) > 0}

    def reset(self):
        # Restore inventory from the previous death before Malmo gets the new mission
        if self._keep_inventory and self._pending_inventory is not None:
            if self._survival_spec is not None:
                self._survival_spec.set_start_inventory(self._pending_inventory)
            self._pending_inventory = None

        obs = self.env.reset()
        self._prev_inventory = self._read_counts(obs)
        self._rewarded = {item.item_id: 0 for item in self._reward_items}
        self._prev_full_inventory = self._read_full_inventory(obs)
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        self._step_count += 1
        reward = 0.0

        # ── Full-inventory delta → item log ───────────────────────────────────
        full_current = self._read_full_inventory(obs)
        for item_id in set(self._prev_full_inventory) | set(full_current):
            prev_qty = self._prev_full_inventory.get(item_id, 0)
            curr_qty = full_current.get(item_id, 0)
            if curr_qty > prev_qty:
                self._item_log.append({
                    "step":    self._step_count,
                    "item_id": item_id,
                    "delta":   curr_qty - prev_qty,
                    "total":   curr_qty,
                })
        self._prev_full_inventory = full_current

        # ── Save inventory on death for next reset ────────────────────────────
        if done and self._keep_inventory:
            self._pending_inventory = tuple(
                (item_id, qty) for item_id, qty in full_current.items()
            )

        # ── Reward computation (reward items only) ────────────────────────────
        current = self._read_counts(obs)
        for item in self._reward_items:
            prev = self._prev_inventory[item.item_id]
            curr = current[item.item_id]
            new_items = max(0, curr - prev)

            if new_items > 0:
                already = self._rewarded[item.item_id]
                if item.max_qty == float("inf"):
                    reward += item.reward_per_item * new_items
                    self._rewarded[item.item_id] = already + new_items
                else:
                    rewardable = max(0, min(already + new_items, item.max_qty) - already)
                    reward += item.reward_per_item * rewardable
                    self._rewarded[item.item_id] = min(already + new_items, item.max_qty)

            self._prev_inventory[item.item_id] = curr

        return obs, reward, done, info
