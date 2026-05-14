import sys
sys.path.insert(0, '/workspace/vpt')

import numpy as np
import torch as th
import cv2

from lib.tree_util import tree_map

AGENT_RESOLUTION = (128, 128)


def _resize_image(img, target_resolution):
    return cv2.resize(img, target_resolution, interpolation=cv2.INTER_LINEAR)


class VPTActor:
    """Wraps MinecraftAgentPolicy for no-gradient rollout inference."""

    def __init__(self, policy, action_mapper, action_transformer, device):
        self.policy = policy
        self.action_mapper = action_mapper
        self.action_transformer = action_transformer
        self.device = device

    def get_initial_hidden_state(self, batch_size=1):
        return self.policy.initial_state(batch_size)

    def step(self, obs, hidden_state, done=False):
        """
        One inference step.

        Args:
            obs:          MineRL obs dict with 'pov' key, shape (H, W, 3) uint8
            hidden_state: recurrent state from previous step (or initial)
            done:         whether the previous step ended an episode

        Returns:
            env_action:   MineRL-compatible action dict for env.step()
            agent_action: dict{"buttons": Tensor, "camera": Tensor} for buffer
            log_prob:     float
            value:        float
            new_hidden:   detached recurrent state for next step
        """
        # MineRL obs["pov"] ist (H, W, C) bei 640x360 → resize auf 128x128 für Policy
        resized = _resize_image(obs["pov"], AGENT_RESOLUTION)

        agent_input = {"img": th.from_numpy(resized[None]).to(self.device)}
        first = th.from_numpy(np.array((done,))).to(self.device)

        agent_action, new_hidden, result = self.policy.act(
            agent_input, first, hidden_state, stochastic=True
        )

        new_hidden = tree_map(lambda x: x.detach(), new_hidden)

        # Interne VPT-Action → MineRL-kompatibles Dict
        action_np = {
            "buttons": agent_action["buttons"].cpu().numpy(),
            "camera":  agent_action["camera"].cpu().numpy(),
        }
        env_action = self.action_transformer.policy2env(
            self.action_mapper.to_factored(action_np)
        )

        return (
            env_action,
            agent_action,
            result["log_prob"].item(),
            result["vpred"].item(),
            new_hidden,
            resized,        # (128, 128, 3) für den Buffer
        )
