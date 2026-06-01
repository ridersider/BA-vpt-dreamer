"""Concatenate TensorBoard reward logs from multiple runs and plot them."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUNS = [
    "run_1779799847",
    "run_1779805425",
    "run_1779810387",
]
LOGS_DIR = "logs"
TAGS = ["train/total_reward"]
COLORS = ["#2196F3", "#FF9800", "#4CAF50"]
OUTPUT = "logs/combined_reward.png"


def load_scalars(run: str, tag: str):
    ea = EventAccumulator(f"{LOGS_DIR}/{run}/tb/")
    ea.Reload()
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def concat_runs(runs, tag, cumulative=False):
    all_steps, all_values = [], []
    step_offset = 0
    value_offset = 0
    for run in runs:
        data = load_scalars(run, tag)
        steps = [s + step_offset for s, _ in data]
        values = [v + value_offset for _, v in data]
        all_steps.extend(steps)
        all_values.extend(values)
        step_offset = steps[-1]
        if cumulative:
            value_offset = values[-1]
    return all_steps, all_values


def run_boundaries():
    boundaries = [0]
    offset = 0
    for run in RUNS[:-1]:
        data = load_scalars(run, TAGS[0])
        offset += data[-1][0]
        boundaries.append(offset)
    return boundaries


fig, axes = plt.subplots(len(TAGS), 1, figsize=(14, 5 * len(TAGS)), sharex=True)
if len(TAGS) == 1:
    axes = [axes]

boundaries = run_boundaries()

CUMULATIVE_TAGS = {"train/total_reward"}

for ax, tag in zip(axes, TAGS):
    steps, values = concat_runs(RUNS, tag, cumulative=tag in CUMULATIVE_TAGS)
    ax.plot(steps, values, linewidth=1.5, color="#333333", label=tag)

    for i, (run, boundary) in enumerate(zip(RUNS, boundaries)):
        ax.axvline(boundary, color=COLORS[i], linestyle="--", alpha=0.6, linewidth=1)
        ax.axvspan(
            boundary,
            boundaries[i + 1] if i + 1 < len(boundaries) else steps[-1],
            alpha=0.05,
            color=COLORS[i],
        )

    ax.set_ylabel("Reward")
    ax.set_title(tag)
    ax.grid(True, alpha=0.3)

    legend_patches = [
        mpatches.Patch(color=COLORS[i], alpha=0.4, label=run)
        for i, run in enumerate(RUNS)
    ]
    ax.legend(handles=legend_patches, fontsize=8, loc="upper left")

axes[-1].set_xlabel("Step (concatenated)")
fig.suptitle("Combined reward across runs", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150)
print(f"Saved: {OUTPUT}")
plt.show()
