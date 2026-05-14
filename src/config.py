import os
import time
from dataclasses import dataclass, field
from typing import List


# ── Item variant lists ────────────────────────────────────────────────────────
# All inventory keys that map to a single logical item.

LOG_VARIANTS: List[str] = [
    "oak_log",    "birch_log",    "spruce_log",
    "jungle_log", "acacia_log",   "dark_oak_log",
    "oak_wood",   "birch_wood",   "spruce_wood",
    "jungle_wood","acacia_wood",  "dark_oak_wood",
]

PLANK_VARIANTS: List[str] = [
    "oak_planks",    "birch_planks",    "spruce_planks",
    "jungle_planks", "acacia_planks",   "dark_oak_planks",
    "planks",   # fallback for older MineRL versions
]


# ── Reward item ───────────────────────────────────────────────────────────────

@dataclass
class RewardItem:
    """
    One rewarded item.

    item_id         Logical key used in reward tables and tasks.
    display_name    Human-readable label (for logs / TensorBoard).
    max_qty         Maximum quantity that earns reward; float('inf') = uncapped.
    reward_per_item Reward given per newly collected unit (up to max_qty).
    variants        All inventory keys that count toward this item.
                    Leave empty when item_id matches the inventory key directly.
    """
    item_id: str
    display_name: str
    max_qty: float
    reward_per_item: float
    variants: List[str] = field(default_factory=list)


# ── Master reward table ───────────────────────────────────────────────────────
# Source of truth for all rewarded items (based on VPT paper, Table 7).
# To add a new item: append a RewardItem entry here.
# To change the reward: edit max_qty or reward_per_item.
# Tasks reference items by item_id.

MASTER_REWARD_TABLE: List[RewardItem] = [
    # ── Wood & crafting basics ──────────────────────────────────────────
    RewardItem("log",             "Log",              max_qty=8,            reward_per_item=1/8,  variants=LOG_VARIANTS),
    RewardItem("planks",          "Planks",           max_qty=20,           reward_per_item=1/20, variants=PLANK_VARIANTS),
    RewardItem("stick",           "Stick",            max_qty=16,           reward_per_item=1/16),
    RewardItem("crafting_table",  "Crafting Table",   max_qty=1,            reward_per_item=1.0),
    # ── Stone tier ─────────────────────────────────────────────────────
    RewardItem("cobblestone",     "Cobblestone",      max_qty=11,           reward_per_item=1/11),
    RewardItem("wooden_pickaxe",  "Wooden Pickaxe",   max_qty=1,            reward_per_item=1.0),
    RewardItem("stone_pickaxe",   "Stone Pickaxe",    max_qty=1,            reward_per_item=1.0),
    # ── Utility ────────────────────────────────────────────────────────
    RewardItem("furnace",         "Furnace",          max_qty=1,            reward_per_item=1.0),
    RewardItem("coal",            "Coal",             max_qty=5,            reward_per_item=2/5),
    RewardItem("torch",           "Torch",            max_qty=16,           reward_per_item=1/8),
    # ── Iron tier ──────────────────────────────────────────────────────
    RewardItem("iron_ore",        "Iron Ore",         max_qty=3,            reward_per_item=4/3),
    RewardItem("iron_ingot",      "Iron Ingot",       max_qty=3,            reward_per_item=4/3),
    RewardItem("iron_pickaxe",    "Iron Pickaxe",     max_qty=1,            reward_per_item=4.0),
    # ── Diamond tier (uncapped) ─────────────────────────────────────────
    RewardItem("diamond",         "Diamond",          max_qty=float("inf"), reward_per_item=8/3),
    RewardItem("diamond_pickaxe", "Diamond Pickaxe",  max_qty=float("inf"), reward_per_item=8.0),
]

_REWARD_INDEX: dict = {item.item_id: item for item in MASTER_REWARD_TABLE}


def get_reward_item(item_id: str) -> RewardItem:
    """Return the RewardItem for item_id. Raises KeyError if not in MASTER_REWARD_TABLE."""
    return _REWARD_INDEX[item_id]


# ── Task definition ───────────────────────────────────────────────────────────

@dataclass
class TaskSpec:
    """
    A single training task.

    name               Short identifier used for filenames / logging.
    display_name       Human-readable label.
    item_reward_keys   Items from MASTER_REWARD_TABLE to reward during this task.
    goal_key           Item whose acquisition completes this task, OR "reward_based"
                       for obtain tasks that complete when max_reward is accumulated.
    action_reward_key  Action key in env_action dict whose press toggles the UI open/closed.
                       Empty string means no action reward.
    action_reward      Reward given per step while the UI is open.
    """
    name: str
    display_name: str
    item_reward_keys: List[str]
    goal_key: str
    action_reward_key: str = ""
    action_reward: float = 0.01
    disable_movement: bool = False

    @property
    def task_type(self) -> str:
        """'obtain' for reward-based completion, 'craft' for item-acquisition completion."""
        return "obtain" if self.goal_key == "reward_based" else "craft"

    @property
    def reward_items(self) -> List[RewardItem]:
        return [get_reward_item(k) for k in self.item_reward_keys]

    @property
    def max_reward(self) -> float:
        """Maximum reward achievable for this task (used by obtain tasks as completion threshold)."""
        return sum(
            item.max_qty * item.reward_per_item
            for item in self.reward_items
            if item.max_qty != float("inf")
        )


# ── Curriculum ────────────────────────────────────────────────────────────────
# Ordered list of tasks for sequential training.
# Add, remove, or reorder tasks here to change the curriculum.

TASKS: List[TaskSpec] = [
    TaskSpec(
        name="obtain_wood",
        display_name="Obtain Wood",
        item_reward_keys=["log"],
        goal_key="reward_based",
    ),
    TaskSpec(
        name="craft_wood_items",
        display_name="Craft Wooden Items",
        item_reward_keys=["stick", "crafting_table"],
        goal_key="crafting_table",
        action_reward_key="inventory",  # E opens the 2x2 crafting grid
        action_reward=0.001,
        disable_movement=True,
    ),
    TaskSpec(
        name="craft_wooden_pickaxe",
        display_name="Craft Wooden Pickaxe",
        item_reward_keys=["wooden_pickaxe"],
        goal_key="wooden_pickaxe",
        disable_movement=True,
    ),
    TaskSpec(
        name="obtain_stone",
        display_name="Obtain Stone",
        item_reward_keys=["cobblestone"],
        goal_key="reward_based",
    ),
    TaskSpec(
        name="craft_stone_pickaxe",
        display_name="Craft Stone Pickaxe",
        item_reward_keys=["stone_pickaxe"],
        goal_key="stone_pickaxe",
    ),
]


# ── Training configuration ────────────────────────────────────────────────────

_MODELS = os.environ.get("VPT_MODELS", "/workspace/models")
_LOGS   = os.environ.get("VPT_LOGS",   "/workspace/logs")
_TASK_INVENTORIES = os.environ.get("VPT_TASK_INVENTORIES", "/workspace/task_inventories")


@dataclass(frozen=True)
class PPOConfig:
    # ── Training mode ──────────────────────────────────────────────────
    # "sequential"  train on TASKS curriculum (sequential_env)
    # "single"      train on a flat reward for single_task_items only
    mode: str = "sequential"
    # item_ids to reward in "single" mode (must exist in MASTER_REWARD_TABLE)
    single_task_items: tuple = ("log",)
    # If True, stay on the current task after completion instead of advancing
    train_on_repeat: bool = True

    # ── Rollout ────────────────────────────────────────────────────────
    # "fixed"            collect exactly n_steps, always update
    # "reward_triggered" collect until reward > 0 or max_steps_no_reward
    update_mode: str = "fixed"
    n_steps: int = 256
    max_steps_no_reward: int = 4096

    # ── PPO hyperparameters ────────────────────────────────────────────
    n_epochs: int = 1
    batch_size: int = 64
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 5.0
    target_kl: float = 0.02

    # ── KL regularization vs frozen pretrained policy ──────────────────
    kl_pretrained_coef: float = 0.2
    kl_pretrained_coef_decay: float = 0.9995

    # ── Optimizer ──────────────────────────────────────────────────────
    learning_rate: float = 2e-5
    weight_decay: float = 0.001

    # ── Model paths ────────────────────────────────────────────────────
    # Override via VPT_MODELS env var or pass explicitly to PPOConfig().
    model_path: str = field(default_factory=lambda: os.path.join(_MODELS, "foundation-model-1x.model"))
    weights_path: str = field(default_factory=lambda: os.path.join(_MODELS, "foundation-model-1x.weights"))
    pretrained_weights_path: str = field(default_factory=lambda: os.path.join(_MODELS, "foundation-model-1x.weights"))
    task_weights_dir: str = field(default_factory=lambda: os.path.join(_MODELS, "task_weights"))
    task_inventories_dir: str = field(default_factory=lambda: _TASK_INVENTORIES)

    # ── Curriculum start ───────────────────────────────────────────────
    # start_task:      name of the task to begin from (skips earlier tasks).
    #                  Must match a TaskSpec.name in TASKS. "" = start from first task.
    # start_inventory: items to place in the agent's inventory at episode start.
    #                  Tuple of (minecraft_item_id, quantity) pairs.
    #                  Use actual MC item IDs (e.g. "oak_log", not the aggregated "log").
    #                  Example: (("oak_log", 8), ("stick", 4))
    start_task: str = "craft_wood_items"
    start_inventory: tuple = (("spruce_log", 8),)

    # ── Training length ────────────────────────────────────────────────
    episodes: int = 20
    total_timesteps: int = 1000
    world_seed: int = 42

    # ── Live stream ────────────────────────────────────────────────────
    stream_enabled: bool = True
    stream_port: int = 8080
    stream_fps: int = 40

    # ── Video / TensorBoard logging ────────────────────────────────────
    record_every_n_updates: int = 999999
    video_fps: int = 20
    log_dir: str = field(default_factory=lambda: os.path.join(_LOGS, f"run_{int(time.time())}"))
