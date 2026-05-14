import sys
sys.path.insert(0, '/workspace/vpt')

import numpy as np
import torch as th
import torch.nn.functional as F

from lib.tree_util import tree_map


class PPOUpdater:
    """Runs PPO gradient updates over a filled VPTRolloutBuffer."""

    def __init__(self, policy, optimizer, config, pretrained_policy=None):
        self.policy = policy
        self.optimizer = optimizer
        self.config = config
        self.pretrained_policy = pretrained_policy  # frozen reference; None → use entropy bonus
        self.kl_coef = config.kl_pretrained_coef

    def decay_kl_coef(self):
        """Call once per iteration to decay the KL coefficient."""
        self.kl_coef *= self.config.kl_pretrained_coef_decay

    def _compute_kl_vs_pretrained(self, obs_seq: th.Tensor, first_seq: th.Tensor,
                                   pi_logits_current: dict) -> th.Tensor:
        """
        KL(π_pretrained || π_current) over the full rollout sequence.
        Runs the pretrained policy with the correct LSTM state threading.
        obs_seq:  (1, n, H, W, C)
        first_seq: (1, n) bool — True resets the LSTM hidden state
        pi_logits_current: dict of (n, ...) — already flattened from (1, n, ...)
        """
        with th.no_grad():
            state_in_pt = self.pretrained_policy.initial_state(1)
            (pi_logits_pt, _, _), _ = self.pretrained_policy(
                obs={"img": obs_seq}, first=first_seq, state_in=state_in_pt
            )
            # (1, n, ...) → (n, ...)
            pi_logits_pt_flat = tree_map(lambda x: x[0], pi_logits_pt)

        total_kl = th.tensor(0.0, device=obs_seq.device)
        for key in pi_logits_pt_flat:
            lp = pi_logits_pt_flat[key].float()     # pretrained logits (n, ...)
            lq = pi_logits_current[key].float()     # current policy logits (n, ...)
            log_p = F.log_softmax(lp, dim=-1)
            log_q = F.log_softmax(lq, dim=-1)
            p = log_p.exp()
            total_kl = total_kl + (p * (log_p - log_q)).sum(dim=-1).mean()

        return total_kl

    def update(self, buffer) -> dict:
        """
        PPO update over the full rollout as one ordered sequence (B=1, T=n).

        Processes all n steps in temporal order so the LSTM hidden state is
        threaded correctly — episode boundaries in buffer.dones are converted
        to 'first' flags that reset the hidden state at each new episode.
        This ensures new_log_prob ≈ old_log_prob at the start of each update
        (approx_kl ≈ 0), which was impossible with the previous shuffled
        minibatch approach that used initial_state for every sample.
        """
        cfg = self.config
        n = buffer.pos
        device = next(self.policy.parameters()).device

        # ── Build full-sequence tensors ───────────────────────────────────────
        obs_np = buffer.obs_imgs[:n].astype(np.float32) / 255.0  # (n, H, W, C)
        obs_seq = th.from_numpy(obs_np).to(device).unsqueeze(0)  # (1, n, H, W, C)

        # first[t]=True → reset LSTM hidden state at step t
        # Step 0 always resets; subsequent steps reset after a done.
        first_np = np.zeros(n, dtype=bool)
        first_np[0] = True
        if n > 1:
            first_np[1:] = buffer.dones[:n - 1].astype(bool)
        first_seq = th.from_numpy(first_np).to(device).unsqueeze(0)  # (1, n)

        old_log_probs = th.from_numpy(buffer.log_probs[:n]).to(device)   # (n,)
        advantages    = th.from_numpy(buffer.advantages[:n]).to(device)  # (n,)
        advantages    = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns_seq   = th.from_numpy(buffer.returns[:n]).to(device)     # (n,)

        agent_action = {
            "buttons": th.from_numpy(buffer.buttons[:n]).to(device),  # (n, 1)
            "camera":  th.from_numpy(buffer.cameras[:n]).to(device),  # (n, 1)
        }

        all_policy_losses, all_value_losses, all_entropies, all_kls, all_kl_pt = [], [], [], [], []

        for _ in range(cfg.n_epochs):
            # ── Sequential forward pass — LSTM threads through all n steps ──
            state_in = self.policy.initial_state(1)
            (pi_logits, vpred_raw, _), _ = self.policy(
                obs={"img": obs_seq}, first=first_seq, state_in=state_in
            )
            # pi_logits: dict of (1, n, ...) → treat timesteps as batch dim → (n, ...)
            pi_logits_flat = tree_map(lambda x: x[0], pi_logits)
            # vpred_raw: (1, n, 1)

            new_log_prob = self.policy.get_logprob_of_action(pi_logits_flat, agent_action)  # (n,)
            entropy = self.policy.pi_head.entropy(pi_logits_flat).mean()

            # ── PPO clip loss ─────────────────────────────────────────────────
            ratio = th.exp(new_log_prob - old_log_probs)
            policy_loss = th.max(
                -advantages * ratio,
                -advantages * th.clamp(ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range),
            ).mean()

            # ── Value loss — vpred: (1, n, 1), target: (1, n, 1) ─────────────
            value_loss = self.policy.value_head.loss(
                vpred_raw,
                returns_seq.unsqueeze(0).unsqueeze(-1),
            )

            # ── Regularization: KL vs. pretrained (or entropy fallback) ──────
            if self.pretrained_policy is not None:
                kl_pt = self._compute_kl_vs_pretrained(obs_seq, first_seq, pi_logits_flat)
                loss = policy_loss + cfg.vf_coef * value_loss + self.kl_coef * kl_pt
                all_kl_pt.append(kl_pt.item())
            else:
                loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy
                all_kl_pt.append(0.0)

            self.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
            self.optimizer.step()

            # ── Logging / early stop ──────────────────────────────────────────
            with th.no_grad():
                log_ratio = new_log_prob - old_log_probs
                approx_kl = ((th.exp(log_ratio) - 1) - log_ratio).mean().item()

            all_policy_losses.append(policy_loss.item())
            all_value_losses.append(value_loss.item())
            all_entropies.append(entropy.item())
            all_kls.append(approx_kl)

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        return {
            "policy_loss":   float(np.mean(all_policy_losses)),
            "value_loss":    float(np.mean(all_value_losses)),
            "entropy":       float(np.mean(all_entropies)),
            "approx_kl":     float(np.mean(all_kls)),
            "kl_pretrained": float(np.mean(all_kl_pt)),
        }
