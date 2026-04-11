import sys
import os
sys.path.insert(0, '/workspace/vpt')

import pickle
import gym

import torch as th
from gym3.types import DictType

from agent import ENV_KWARGS
from tree_flat_env import TreeFlatEnv
from lib.action_mapping import CameraHierarchicalMapping
from lib.actions import ActionTransformer
from lib.policy import MinecraftAgentPolicy

from config import PPOConfig
from vpt_actor import VPTActor, LogHarvestEnv
from rollout_buffer import VPTRolloutBuffer
from ppo_updater import PPOUpdater
from video_logger import VideoLogger
from live_stream import LiveStreamer

# ── Constants taken from repos/vpt/agent.py ────────────────────────────────
PI_HEAD_KWARGS = {"temperature": 2.0}

ACTION_TRANSFORMER_KWARGS = dict(
    camera_binsize=2,
    camera_maxval=10,
    camera_mu=10,
    camera_quantization_scheme="mu_law",
)


def load_model_parameters(path_to_model_file):
    """Load policy_kwargs and pi_head_kwargs from a .model file (pickle)."""
    agent_parameters = pickle.load(open(path_to_model_file, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    return policy_kwargs, pi_head_kwargs


def build_policy(config: PPOConfig, device: th.device) -> MinecraftAgentPolicy:
    """Build MinecraftAgentPolicy directly (bypasses MineRLAgent.validate_env)."""
    policy_kwargs, pi_head_kwargs = load_model_parameters(config.model_path)
    action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
    action_space = DictType(**action_mapper.get_action_space_update())

    policy = MinecraftAgentPolicy(
        action_space=action_space,
        policy_kwargs=policy_kwargs,
        pi_head_kwargs=pi_head_kwargs,
    ).to(device)

    policy.load_state_dict(
        th.load(config.weights_path, map_location=device), strict=False
    )
    print(f"[train_ppo] Loaded weights from {config.weights_path}")
    return policy, action_mapper


def main():
    config = PPOConfig()
    device = th.device("cpu")

    # ── Environment ────────────────────────────────────────────────────────
    env = LogHarvestEnv(TreeFlatEnv(**ENV_KWARGS).make())
    print("[train_ppo] MineRL LogHarvestEnv created.")

    # ── Policy ─────────────────────────────────────────────────────────────
    policy, action_mapper = build_policy(config, device)
    action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

    # ── Training components ────────────────────────────────────────────────
    actor = VPTActor(policy, action_mapper, action_transformer, device)
    buffer = VPTRolloutBuffer(
        n_steps=config.n_steps,
        image_shape=(128, 128, 3),
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        device=device,
    )
    optimizer = th.optim.Adam(
        policy.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    updater = PPOUpdater(policy, optimizer, config)
    streamer = LiveStreamer(port=config.stream_port, fps=config.stream_fps)
    streamer.start()
    video_logger = VideoLogger(
        log_dir=config.log_dir,
        fps=config.video_fps,
        record_every=config.record_every_n_updates,
    )

    # ── Training loop ──────────────────────────────────────────────────────
    obs = env.reset()
    hidden = actor.get_initial_hidden_state()
    done = False
    total_steps = 0
    update_num = 0
    total_reward = 0.0

    print(f"[train_ppo] Starting training for {config.total_timesteps} steps.")

    while total_steps < config.total_timesteps:
        buffer.reset()

        # ── Rollout collection (no gradients) ──────────────────────────────
        policy.eval()
        rollout_reward = 0.0
        video_logger.start_rollout(update_num)
        for _ in range(config.n_steps):
            video_logger.add_frame(obs["pov"])
            streamer.push_frame(obs["pov"])

            env_action, agent_action, log_prob, value, hidden, obs_128 = actor.step(
                obs, hidden, done
            )
            next_obs, reward, done, _ = env.step(env_action)

            buffer.add(obs_128, agent_action, reward, done, value, log_prob)

            rollout_reward += reward
            total_reward += reward
            obs = next_obs
            total_steps += 1

            if done:
                obs = env.reset()
                hidden = actor.get_initial_hidden_state()

        # ── Compute GAE ────────────────────────────────────────────────────
        _, _, _, last_value, _, _ = actor.step(obs, hidden, done)
        buffer.finalize(last_value, done)

        # ── PPO update ─────────────────────────────────────────────────────
        if rollout_reward == 0.0:
            print(f"[Update {update_num + 1:4d}] Skipping update (no reward collected).")
            update_num += 1
            continue

        policy.train()
        metrics = updater.update(buffer)

        # Reset world after each update so the agent always starts fresh
        #obs = env.reset()
        #hidden = actor.get_initial_hidden_state()
        #done = False

        update_num += 1
        print(
            f"[Update {update_num:4d} | Steps {total_steps:8d}] "
            f"policy_loss={metrics['policy_loss']:.4f}  "
            f"value_loss={metrics['value_loss']:.4f}  "
            f"entropy={metrics['entropy']:.4f}  "
            f"approx_kl={metrics['approx_kl']:.4f}  "
            f"rollout_reward={rollout_reward:.1f}  total_reward={total_reward:.1f}"
        )

        # ── Logging + optional video recording ────────────────────────────
        video_logger.log_metrics(update_num, metrics)
        video_logger.maybe_save_video(update_num)

        # ── Checkpoint ─────────────────────────────────────────────────────
        if update_num % config.checkpoint_every_n_updates == 0:
            ckpt_path = os.path.join(config.log_dir, f"checkpoint_{update_num:05d}.weights")
            th.save(policy.state_dict(), ckpt_path)
            print(f"[train_ppo] Checkpoint saved to {ckpt_path}")

    # ── Save final weights ─────────────────────────────────────────────────
    th.save(policy.state_dict(), config.out_weights_path)
    print(f"[train_ppo] Saved weights to {config.out_weights_path}")

    video_logger.close()
    env.close()


if __name__ == "__main__":
    main()
