import sys
sys.path.insert(0, '/workspace/vpt')

import numpy as np
import torch as th

from lib.tree_util import tree_map


class PPOUpdater:
    """Runs PPO gradient updates over a filled VPTRolloutBuffer."""

    def __init__(self, policy, optimizer, config):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config

    def update(self, buffer) -> dict:
        cfg = self.config
        all_policy_losses = []
        all_value_losses = []
        all_entropies = []
        all_kls = []
        stop_early = False

        for _ in range(cfg.n_epochs):
            if stop_early:
                break
            for batch in buffer.get(cfg.batch_size):
                # ── Build agent input (B, 1, H, W, C) ──────────────────────
                obs_np = batch["obs_imgs"].astype(np.float32) / 255.0  # (B, H, W, C)
                B = obs_np.shape[0]
                device = next(self.policy.parameters()).device
                obs_t = th.from_numpy(obs_np).to(device)
                obs_t = obs_t.unsqueeze(1)  # (B, 1, H, W, C)

                first = th.zeros(B, dtype=th.bool).to(device)   # (B,)
                first_t = first.unsqueeze(1)                      # (B, 1)

                state_in = self.policy.initial_state(B)

                # ── Forward pass (with gradients) ───────────────────────────
                (pi_logits, vpred_raw, _), _ = self.policy(
                    obs={"img": obs_t}, first=first_t, state_in=state_in
                )
                # pi_logits: distribution dict at (B, 1, ...)
                # vpred_raw: (B, 1, 1) — raw normalized value before denormalize

                # Squeeze T=1 dimension from pi_logits
                pi_logits_0 = tree_map(lambda x: x[:, 0], pi_logits)
                vpred_sq = vpred_raw[:, 0:1, :]  # (B, 1, 1) — NormalizeEwma needs (B, T, out)

                # ── Reconstruct stored actions ───────────────────────────────
                agent_action = {
                    "buttons": batch["buttons"].to(device),  # (B, 1)
                    "camera": batch["cameras"].to(device),   # (B, 1)
                }

                # ── Log-prob + entropy of stored actions ─────────────────────
                new_log_prob = self.policy.get_logprob_of_action(pi_logits_0, agent_action)
                entropy = self.policy.pi_head.entropy(pi_logits_0).mean()

                # ── PPO clip loss ─────────────────────────────────────────────
                old_log_probs = batch["old_log_probs"]
                advantages = batch["advantages"]

                # Normalize advantages within the minibatch
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(new_log_prob - old_log_probs)
                policy_loss_1 = -advantages * ratio
                policy_loss_2 = -advantages * th.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range)
                policy_loss = th.max(policy_loss_1, policy_loss_2).mean()

                # ── Value loss (ScaledMSEHead: normalized pred, denorm target) ─
                value_loss = self.policy.value_head.loss(vpred_sq, batch["returns"].unsqueeze(1).unsqueeze(2))

                # ── Combined loss ─────────────────────────────────────────────
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                # ── Logging / early stop ──────────────────────────────────────
                with th.no_grad():
                    log_ratio = new_log_prob - old_log_probs
                    approx_kl = ((th.exp(log_ratio) - 1) - log_ratio).mean().item()

                all_policy_losses.append(policy_loss.item())
                all_value_losses.append(value_loss.item())
                all_entropies.append(entropy.item())
                all_kls.append(approx_kl)

                if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                    stop_early = True
                    break

        return {
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "entropy": float(np.mean(all_entropies)),
            "approx_kl": float(np.mean(all_kls)),
        }
