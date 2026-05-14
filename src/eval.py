"""
Evaluation script — run a trained agent, record a video, and log item collection.

Usage
-----
    python eval.py                        # uses config defaults
    python eval.py --weights /path/to/model.weights
    python eval.py --steps 2000
    python eval.py --out /workspace/logs/eval_run
    python eval.py --no_keep_inventory    # drop items on death (default: keep)
    python eval.py --no_video             # skip video recording

Environment variables
---------------------
    VPT_MODELS   directory with .model / .weights files  (default /workspace/models)

Outputs
-------
    <out_dir>/eval.mp4       video of the run (unless --no_video)
    <out_dir>/item_log.csv   step,item_id,delta,total — written live during the run
"""

import sys
import os
import argparse
import csv
from collections import defaultdict

sys.path.insert(0, '/workspace/vpt')

import cv2
import torch as th

from agent import ENV_KWARGS
from config import PPOConfig
from training_utils import build_policy, make_action_transformer
from envs.survival_env import SurvivalEnv, SurvivalRewardEnv
from vpt_actor import VPTActor


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained VPT agent")
    p.add_argument("--weights",           default=None,  help="Path to .weights file")
    p.add_argument("--steps",             default=100000,  type=int)
    p.add_argument("--out",               default=None,  help="Output directory (default: <log_dir>)")
    p.add_argument("--fps",               default=20,    type=int)
    p.add_argument("--no_keep_inventory", action="store_true",
                   help="Drop items on death (default: inventory is kept)")
    p.add_argument("--no_video",          action="store_true",
                   help="Skip video recording")
    return p.parse_args()


def _print_summary(csv_path: str) -> None:
    totals = defaultdict(int)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            totals[row["item_id"]] += int(row["delta"])
    print("[eval] Items collected:")
    for item_id, qty in sorted(totals.items(), key=lambda x: -x[1]):
        print(f"         {item_id:30s} {qty:4d}")


def main():
    args = parse_args()
    config = PPOConfig()
    device = th.device("cuda" if th.cuda.is_available() else "cpu")

    weights = args.weights or config.weights_path
    out_dir = args.out or config.log_dir
    os.makedirs(out_dir, exist_ok=True)

    keep_inventory = not args.no_keep_inventory

    # Build env directly (not via make_env) to keep the spec reference
    # needed for inventory restoration on death.
    spec = SurvivalEnv(**ENV_KWARGS)
    base = spec.make()
    env  = SurvivalRewardEnv(base, keep_inventory=keep_inventory, survival_spec=spec)

    policy, action_mapper = build_policy(config, device, weights_path=weights)
    actor = VPTActor(policy, action_mapper, make_action_transformer(), device)

    # ── Video writer (optional) ───────────────────────────────────────────────
    video = None
    if not args.no_video:
        video_path = os.path.join(out_dir, "eval.mp4")
        video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                args.fps, (128, 128))

    # ── Item log: open CSV once and append rows live during the run ───────────
    csv_path = os.path.join(out_dir, "item_log.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=["step", "item_id", "delta", "total"])
    csv_writer.writeheader()
    csv_file.flush()
    log_cursor = 0  # index into env.item_log up to which we've already written

    policy.eval()
    obs = env.reset()
    hidden = actor.get_initial_hidden_state()
    done = False
    total_reward = 0.0
    death_count  = 0
    bar_width    = 40

    print(f"[eval] Running {args.steps} steps  weights={weights}"
          f"  keep_inventory={keep_inventory}  video={not args.no_video}")
    print(f"[eval] Item log → {csv_path}")

    for step in range(args.steps):
        env_action, _, _, _, hidden, frame = actor.step(obs, hidden, done)
        obs, reward, done, info = env.step(env_action)

        if video is not None:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        total_reward += reward

        # Write any new item log entries that appeared this step
        new_entries = env.item_log[log_cursor:]
        if new_entries:
            csv_writer.writerows(new_entries)
            csv_file.flush()
            log_cursor += len(new_entries)

        filled = int(bar_width * min(max(total_reward, 0), bar_width) / bar_width)
        bar = "#" * filled + "-" * (bar_width - filled)
        print(f"\rstep {step+1:5d}  reward={reward:+.3f}  total=[{bar}] {total_reward:.2f}"
              f"  deaths={death_count}", end="", flush=True)

        if done:
            death_count += 1
            obs = env.reset()
            hidden = actor.get_initial_hidden_state()
            done = False

    print(f"\n[eval] Total reward: {total_reward:.2f}  deaths: {death_count}")

    csv_file.close()
    env.close()

    if video is not None:
        video.release()
        print(f"[eval] Video    → {video_path}")

    _print_summary(csv_path)


if __name__ == "__main__":
    main()
