"""Shared helpers: policy construction and environment factory."""

import sys
import os
import pickle
from typing import Optional

sys.path.insert(0, '/workspace/vpt')

import torch as th
from gym3.types import DictType

from agent import ENV_KWARGS
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy

from config import PPOConfig
from envs.survival_env import SurvivalEnv, SurvivalRewardEnv
from envs.sequential_env import SequentialRewardEnv
from envs.navigation_env import NavigationRewardEnv


ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


def load_model_parameters(path: str) -> tuple:
    agent_parameters = pickle.load(open(path, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def find_latest_weights(task_idx: int, task_weights_dir: str) -> Optional[str]:
    """Return path to the latest saved weights for task_idx, or None if not found."""
    from config import TASKS
    task = TASKS[task_idx]
    path = os.path.join(task_weights_dir, f"task_{task_idx:02d}_{task.name}_latest.weights")
    return path if os.path.exists(path) else None


def build_policy(config: PPOConfig, device: th.device,
                 weights_path: str = None) -> tuple:
    """Build and load a MinecraftAgentPolicy. Returns (policy, action_mapper)."""
    policy_kwargs, pi_head_kwargs = load_model_parameters(config.model_path)
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = DictType(**action_mapper.get_action_space_update())

    policy = MinecraftAgentPolicy(
        action_space=action_space,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
    ).to(device)

    path = weights_path or config.weights_path
    policy.load_state_dict(th.load(path, map_location=device), strict=False)
    print(f"[build_policy] Loaded weights from {path}")
    return policy, action_mapper


def build_frozen_policy(config: PPOConfig, device: th.device):
    """Return a frozen pretrained policy for KL regularization, or None."""
    if not os.path.exists(config.pretrained_weights_path):
        print(f"[build_frozen_policy] WARNING: {config.pretrained_weights_path} not found "
              f"— falling back to entropy bonus.")
        return None
    policy, _ = build_policy(config, device,
                              weights_path=config.pretrained_weights_path)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad_(False)
    print(f"[build_frozen_policy] Frozen policy loaded (KL coef={config.kl_pretrained_coef})")
    return policy


def suppress_movement_in_policy(policy, bias_value: float = -10.0) -> None:
    """
    Set large negative biases for all button combinations that contain movement actions
    (forward/back/left/right/sprint/sneak), so the policy starts with near-zero
    probability of moving. Useful for craft tasks where movement is disabled.

    The buttons pi_head is a CategoricalActionHead over BUTTONS_COMBINATIONS.
    Movement groups are at fixed positions in the combination tuple:
      index 1 = fore_back, index 2 = left_right, index 3 = sprint_sneak

    A gradient hook is registered to keep these biases frozen at bias_value
    throughout training — without it the kl_pretrained loss would pull them
    back toward the pretrained (movement-enabled) values.
    """
    combos = CameraHierarchicalMapping.BUTTONS_COMBINATIONS
    movement_positions = {1, 2, 3}  # fore_back, left_right, sprint_sneak in combo tuple
    movement_indices = [
        i for i, combo in enumerate(combos)
        if isinstance(combo, tuple) and any(combo[p] != "none" for p in movement_positions)
    ]
    bias = policy.pi_head["buttons"].linear_layer.bias
    with th.no_grad():
        bias[movement_indices] = bias_value

    # Freeze: zero gradients for movement indices after every backward pass.
    # Without this, the kl_pretrained loss pulls these biases back toward the
    # pretrained values, gradually re-enabling movement.
    idx_t = th.tensor(movement_indices, dtype=th.long)
    def _freeze_movement_grads(grad):
        g = grad.clone()
        g[idx_t] = 0.0
        return g
    bias.register_hook(_freeze_movement_grads)

    print(f"[suppress_movement] bias={bias_value}, frozen {len(movement_indices)}/{len(combos)} movement combos")


def make_action_transformer() -> ActionTransformer:
    return ActionTransformer(**ACTION_TRANSFORMER_KWARGS)


def make_env(config: PPOConfig):
    """
    Build the training environment for the configured mode.

    "sequential"  SequentialRewardEnv  — TASKS curriculum from config.py
    "single"      SurvivalRewardEnv    — flat reward for config.single_task_items
    "navigation"  NavigationRewardEnv  — distance-delta reward to random XZ target
    """
    from config import TASKS

    if config.mode == "navigation":
        nav_spec = SurvivalEnv(**ENV_KWARGS)
        base = nav_spec.make()
        env = NavigationRewardEnv(
            base,
            target_radius=config.nav_target_radius,
            success_radius=config.nav_success_radius,
            reward_scale=config.nav_reward_scale,
            success_bonus=config.nav_success_bonus,
            log_dir=config.log_dir,
            survival_spec=nav_spec,
        )
        print(f"[make_env] mode=navigation  target_radius={config.nav_target_radius}  "
              f"success_radius={config.nav_success_radius}  "
              f"reward_scale={config.nav_reward_scale}  "
              f"success_bonus={config.nav_success_bonus}  "
              f"target_tower=red_wool_y1-255")
        return env

    base = SurvivalEnv(start_inventory=config.start_inventory, **ENV_KWARGS).make()

    if config.mode == "sequential":
        start_task_idx = 0
        if config.start_task:
            names = [t.name for t in TASKS]
            if config.start_task not in names:
                raise ValueError(f"start_task '{config.start_task}' not found in TASKS. "
                                 f"Available: {names}")
            start_task_idx = names.index(config.start_task)

        env = SequentialRewardEnv(base, start_task_idx=start_task_idx, repeat=config.train_on_repeat)
        env.seed(config.world_seed)
        print(f"[make_env] mode=sequential  start_task={TASKS[start_task_idx].display_name}"
              f"  seed={config.world_seed}  repeat={config.train_on_repeat}")
        if config.start_inventory:
            print(f"[make_env] start_inventory={list(config.start_inventory)}")
    else:
        env = SurvivalRewardEnv(base, item_ids=list(config.single_task_items))
        print(f"[make_env] mode=single  items={list(config.single_task_items)}")

    return env
