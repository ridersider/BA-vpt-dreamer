import sys
sys.path.insert(0, '/workspace/vpt')

import os
import pickle
import numpy as np
import cv2
import gym

from minerl.herobraine.env_specs.human_survival_specs import HumanSurvival
from agent import MineRLAgent, ENV_KWARGS


class LogHarvestEnv(gym.Wrapper):
    """
    HumanSurvival + Reward für Holzsammeln.
    Reward = Anzahl neu gesammelter Logs pro Step (Delta im Inventory).
    validate_env() von MineRLAgent sieht durch den Wrapper hindurch,
    da gym.Wrapper Attribute an die innere Env weiterleitet.
    """
    def __init__(self, env):
        super().__init__(env)
        self._prev_logs = 0

    def reset(self):
        obs = self.env.reset()
        self._prev_logs = int(obs["inventory"]["log"])
        return obs

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        logs = int(obs["inventory"]["log"])
        reward = float(logs - self._prev_logs)
        self._prev_logs = logs
        return obs, reward, done, info

MODEL_PATH   = "/workspace/models/foundation-model-1x.model"
WEIGHTS_PATH = "/workspace/models/foundation-model-1x.weights"
VIDEO_PATH   = "/workspace/logs/episode.mp4"


def main():
    print("Loading VPT model...")
    agent_parameters = pickle.load(open(MODEL_PATH, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])

    print("Starting MineRL environment...")
    env = HumanSurvival(**ENV_KWARGS).make()
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(WEIGHTS_PATH)

    obs = env.reset()

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
    video = cv2.VideoWriter(VIDEO_PATH, cv2.VideoWriter_fourcc(*"mp4v"), 20, (128, 128))

    accumulated_reward = 0.0
    reward_history = []
    bar_width = 40

    print("Running agent loop...")
    for step in range(1000):
        action = agent.get_action(obs)
        obs, reward, done, info = env.step(action)

        # Video-Frame schreiben: MineRL gibt obs["pov"] in (H, W, 3) RGB
        frame = cv2.resize(obs["pov"], (128, 128), interpolation=cv2.INTER_LINEAR)
        video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Akkumulierten Reward tracken
        accumulated_reward += reward
        reward_history.append(accumulated_reward)

        max_r = max(reward_history) if max(reward_history) > 0 else 1
        filled = int(bar_width * accumulated_reward / max_r)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\rstep {step:4d}  reward={reward:+.2f}  accumulated=[{bar}] {accumulated_reward:.2f}", end="", flush=True)

        if done:
            print()
            obs = env.reset()
            agent.reset()
            accumulated_reward = 0.0
            reward_history.clear()

    print()
    video.release()
    env.close()
    print(f"Video gespeichert: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
