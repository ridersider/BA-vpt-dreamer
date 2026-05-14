"""
Single entry point for PPO training.

Quick start
-----------
    python train.py                              # sequential curriculum, defaults
    python train.py --mode single                # single-task log harvesting
    python train.py --update_mode reward_triggered --total_timesteps 200000
    python train.py --mode single --single_task_items log planks

Environment variables
---------------------
    VPT_MODELS   directory containing .model and .weights files  (default /workspace/models)
    VPT_LOGS     directory for TensorBoard logs and checkpoints  (default /workspace/logs)
"""

import sys
import os
import json
import argparse

sys.path.insert(0, '/workspace/vpt')

import torch as th

from config import PPOConfig, TASKS
from training_utils import (
    build_policy, build_frozen_policy, make_action_transformer, make_env,
    find_latest_weights, suppress_movement_in_policy,
)
from vpt_actor import VPTActor
from rollout_buffer import VPTRolloutBuffer
from ppo_updater import PPOUpdater
from video_logger import VideoLogger
from live_stream import LiveStreamer


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train VPT agent with PPO")
    p.add_argument("--mode",              default=None,  choices=["sequential", "single"],
                   help="Training mode (default: sequential)")
    p.add_argument("--single_task_items", default=None,  nargs="+",
                   help="Items to reward in single mode, e.g. --single_task_items log planks")
    p.add_argument("--update_mode",       default=None,  choices=["fixed", "reward_triggered"])
    p.add_argument("--n_steps",           default=None,  type=int)
    p.add_argument("--total_timesteps",   default=None,  type=int)
    p.add_argument("--episodes",           default=None,  type=int,
                   help="Stop after this many episodes (0 = use total_timesteps only)")
    p.add_argument("--learning_rate",     default=None,  type=float)
    p.add_argument("--task_weights_dir",  default=None,
                   help="Directory for per-task weight files (overrides config.task_weights_dir)")
    p.add_argument("--log_dir",           default=None)
    p.add_argument("--stream_port",       default=None,  type=int)
    p.add_argument("--stream_enabled",    default=None,  type=lambda x: x.lower() != "false")
    return p.parse_args()


def config_from_args(args) -> PPOConfig:
    overrides = {}
    if args.mode                is not None: overrides["mode"]              = args.mode
    if args.single_task_items   is not None: overrides["single_task_items"] = tuple(args.single_task_items)
    if args.update_mode         is not None: overrides["update_mode"]       = args.update_mode
    if args.n_steps             is not None: overrides["n_steps"]           = args.n_steps
    if args.total_timesteps     is not None: overrides["total_timesteps"]   = args.total_timesteps
    if args.episodes            is not None: overrides["episodes"]          = args.episodes
    if args.learning_rate       is not None: overrides["learning_rate"]     = args.learning_rate
    if args.task_weights_dir    is not None: overrides["task_weights_dir"]  = args.task_weights_dir
    if args.log_dir             is not None: overrides["log_dir"]           = args.log_dir
    if args.stream_port         is not None: overrides["stream_port"]       = args.stream_port
    if args.stream_enabled      is not None: overrides["stream_enabled"]    = args.stream_enabled
    return PPOConfig(**overrides)


# ── Training loop ─────────────────────────────────────────────────────────────

def _collect_fixed(config, env, actor, buffer, video_logger, streamer,
                   obs, hidden, done, total_steps):
    rollout_reward = 0.0
    completed_task_idx = None
    inventory_snapshot = {}
    info = {}
    for _ in range(config.n_steps):
        video_logger.add_frame(obs["pov"])
        streamer.push_frame(obs["pov"])

        env_action, agent_action, log_prob, value, hidden, obs_128 = actor.step(obs, hidden, done)
        next_obs, reward, done, info = env.step(env_action)
        buffer.add(obs_128, agent_action, reward, done, value, log_prob)

        if info.get("task_complete") and completed_task_idx is None:
            completed_task_idx = (
                info["task_idx"] if info.get("all_tasks_complete")
                else info["task_idx"] - 1
            )
            inventory_snapshot = info.get("inventory_snapshot", {})

        rollout_reward += reward
        total_steps += 1
        obs = next_obs

        if done:
            obs, hidden, done = _handle_done(env, actor, info, total_steps)

    return obs, hidden, done, total_steps, rollout_reward, info, completed_task_idx, inventory_snapshot


def _collect_reward_triggered(config, env, actor, buffer, video_logger, streamer,
                               obs, hidden, done, total_steps):
    rollout_reward = 0.0
    completed_task_idx = None
    inventory_snapshot = {}
    info = {}
    while not buffer.full and total_steps < config.total_timesteps:
        video_logger.add_frame(obs["pov"])
        streamer.push_frame(obs["pov"])

        env_action, agent_action, log_prob, value, hidden, obs_128 = actor.step(obs, hidden, done)
        next_obs, reward, done, info = env.step(env_action)
        buffer.add(obs_128, agent_action, reward, done, value, log_prob)

        if info.get("task_complete") and completed_task_idx is None:
            completed_task_idx = (
                info["task_idx"] if info.get("all_tasks_complete")
                else info["task_idx"] - 1
            )
            inventory_snapshot = info.get("inventory_snapshot", {})

        rollout_reward += reward
        total_steps += 1
        obs = next_obs

        if done:
            obs, hidden, done = _handle_done(env, actor, info, total_steps)

        if rollout_reward > 0.0:
            break

    return obs, hidden, done, total_steps, rollout_reward, info, completed_task_idx, inventory_snapshot


def _handle_done(env, actor, info, total_steps):
    if info.get("all_tasks_complete"):
        print(f"  [Step {total_steps:7d}] All tasks complete!")
    if info.get("player_died"):
        print(f"  [Step {total_steps:7d}] Agent died — resetting.")
    return env.reset(), actor.get_initial_hidden_state(), False


def _log_task_events(info: dict, prev_task_idx: int, step: int) -> None:
    if not info.get("task_complete"):
        return
    if info.get("all_tasks_complete"):
        print(f"  [Step {step:7d}] ✓ Final task '{TASKS[prev_task_idx].display_name}' complete!")
    else:
        print(f"  [Step {step:7d}] ✓ '{TASKS[prev_task_idx].display_name}' done "
              f"→ '{TASKS[info['task_idx']].display_name}'")


def _save_task_checkpoint(completed_task_idx, policy, out_dir: str, step: int):
    if completed_task_idx is None:
        return
    task_name = TASKS[completed_task_idx].name
    path = os.path.join(out_dir, f"task_{completed_task_idx:02d}_{task_name}.weights")
    th.save(policy.state_dict(), path)
    print(f"  [Step {step:7d}] Checkpoint saved → {path}")


def _save_current_weights(task_idx, policy, out_dir: str):
    task = TASKS[task_idx]
    path = os.path.join(out_dir, f"task_{task_idx:02d}_{task.name}_latest.weights")
    th.save(policy.state_dict(), path)


def _maybe_load_task_weights(task_idx, policy, device, task_weights_dir: str) -> bool:
    path = find_latest_weights(task_idx, task_weights_dir)
    if path is None:
        return False
    policy.load_state_dict(th.load(path, map_location=device), strict=False)
    print(f"  [Weights] Loaded task {task_idx} weights from {path}")
    return True


def _save_inventory_snapshot(completed_task_idx, inventory_snapshot: dict,
                              out_dir: str, step: int):
    if completed_task_idx is None or not inventory_snapshot:
        return
    task = TASKS[completed_task_idx]
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"task_{completed_task_idx:02d}_{task.name}_inventory.json")
    data = {
        "task_name": task.name,
        "task_idx": completed_task_idx,
        "completed_at_step": step,
        "inventory": inventory_snapshot,
        "start_inventory": [[k, v] for k, v in inventory_snapshot.items()],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  [Step {step:7d}] Inventory snapshot → {path}")


def train(config: PPOConfig) -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"[train] device={device}  mode={config.mode}  update_mode={config.update_mode}")

    env = make_env(config)

    policy, action_mapper = build_policy(config, device)
    frozen_policy = build_frozen_policy(config, device)
    action_transformer = make_action_transformer()

    # Resume from saved checkpoint if available for the start task
    start_task_idx = 0
    if config.mode == "sequential" and config.start_task:
        names = [t.name for t in TASKS]
        if config.start_task in names:
            start_task_idx = names.index(config.start_task)
    _maybe_load_task_weights(start_task_idx, policy, device, config.task_weights_dir)

    if config.mode == "sequential" and TASKS[start_task_idx].disable_movement:
        suppress_movement_in_policy(policy)

    actor = VPTActor(policy, action_mapper, action_transformer, device)
    buffer_capacity = config.n_steps if config.update_mode == "fixed" else config.max_steps_no_reward
    buffer = VPTRolloutBuffer(
        capacity=buffer_capacity,
        image_shape=(128, 128, 3),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device=device,
    )
    optimizer = th.optim.Adam(policy.parameters(), lr=config.learning_rate,
                               weight_decay=config.weight_decay)
    updater = PPOUpdater(policy, optimizer, config, pretrained_policy=frozen_policy)

    streamer = LiveStreamer(port=config.stream_port, fps=config.stream_fps)
    if config.stream_enabled:
        streamer.start()
    video_logger = VideoLogger(log_dir=config.log_dir, fps=config.video_fps,
                                record_every=config.record_every_n_updates)

    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.task_weights_dir, exist_ok=True)

    obs = env.reset()
    hidden = actor.get_initial_hidden_state()
    done = False
    total_steps = 0
    episode_count = 0
    episode_steps = 0   # steps within the current episode (only used when episodes > 0)
    update_num = 0
    total_reward = 0.0
    prev_task_idx = getattr(env, "task_idx", 0)

    if config.episodes > 0:
        print(f"[train] Starting — {config.episodes} episodes × {config.total_timesteps:,} steps/episode.")
    else:
        print(f"[train] Starting — {config.total_timesteps:,} total steps.")
    if config.mode == "sequential":
        print(f"[train] Tasks: {[t.display_name for t in TASKS]}")

    def _not_done() -> bool:
        if config.episodes > 0:
            return episode_count < config.episodes
        return total_steps < config.total_timesteps

    while _not_done():
        buffer.reset()
        policy.eval()
        video_logger.start_rollout(update_num)
        info = {}

        if config.update_mode == "fixed":
            obs, hidden, done, total_steps, rollout_reward, info, completed_task_idx, inventory_snapshot = _collect_fixed(
                config, env, actor, buffer, video_logger, streamer,
                obs, hidden, done, total_steps,
            )
            episode_steps += config.n_steps
        else:
            obs, hidden, done, total_steps, rollout_reward, info, completed_task_idx, inventory_snapshot = _collect_reward_triggered(
                config, env, actor, buffer, video_logger, streamer,
                obs, hidden, done, total_steps,
            )

        total_reward += rollout_reward

        _log_task_events(info, prev_task_idx, total_steps)
        _save_task_checkpoint(completed_task_idx, policy, config.task_weights_dir, total_steps)
        _save_inventory_snapshot(completed_task_idx, inventory_snapshot, config.task_inventories_dir, total_steps)
        new_task_idx = info.get("task_idx", prev_task_idx)
        if completed_task_idx is not None and new_task_idx != prev_task_idx:
            _maybe_load_task_weights(new_task_idx, policy, device, config.task_weights_dir)
            if TASKS[new_task_idx].disable_movement:
                suppress_movement_in_policy(policy)
        prev_task_idx = new_task_idx

        # GAE
        _, _, _, last_value, _, _ = actor.step(obs, hidden, done)
        buffer.finalize(last_value, done)

        if config.update_mode == "reward_triggered" and rollout_reward == 0.0:
            print(f"[Update {update_num + 1:4d}] Skipping (buffer full, no reward).")
            update_num += 1
            continue

        # PPO update
        policy.train()
        metrics = updater.update(buffer)
        updater.decay_kl_coef()
        update_num += 1
        _save_current_weights(info.get("task_idx", prev_task_idx), policy, config.task_weights_dir)

        ep_label = f"ep={episode_count + 1}/{config.episodes}  " if config.episodes > 0 else ""
        task_label = (
            f"task={info.get('task_idx', 0)}:{TASKS[info['task_idx']].name}"
            if config.mode == "sequential" and "task_idx" in info
            else ""
        )
        print(
            f"[Update {update_num:4d} | Steps {total_steps:8d} | {ep_label}{task_label}] "
            f"policy_loss={metrics['policy_loss']:.4f}  "
            f"value_loss={metrics['value_loss']:.4f}  "
            f"approx_kl={metrics['approx_kl']:.4f}  "
            f"kl_pt={metrics['kl_pretrained']:.4f}  "
            f"kl_coef={updater.kl_coef:.4f}  "
            f"rollout_r={rollout_reward:.2f}  total_r={total_reward:.1f}"
        )

        video_logger.log_metrics(update_num, {
            **metrics,
            "rollout_reward": rollout_reward,
            "total_reward": total_reward,
            "task_idx": info.get("task_idx", 0),
        })
        video_logger.maybe_save_video(update_num)

        if done and info.get("all_tasks_complete"):
            break

        # Episode boundary: force-reset after total_timesteps steps per episode
        if config.episodes > 0 and episode_steps >= config.total_timesteps:
            obs = env.reset()
            hidden = actor.get_initial_hidden_state()
            done = False
            episode_count += 1
            episode_steps = 0
            print(f"[Episode {episode_count}/{config.episodes}] Reset after {config.total_timesteps} steps.")

    video_logger.close()
    env.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    config = config_from_args(args)
    train(config)


if __name__ == "__main__":
    main()
