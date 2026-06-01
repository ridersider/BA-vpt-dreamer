import pandas as pd
import matplotlib.pyplot as plt
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "logs/eval_run/reward_log.csv"
df = pd.read_csv(path)

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(df["step"], df["total_reward"], linewidth=1.5, label="Cumulative reward")
ax.bar(df["step"], df["reward"], alpha=0.3, width=50, label="Reward per step")

ax.set_xlabel("Step")
ax.set_ylabel("Reward")
ax.set_title(f"Reward over time — {path}")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = path.replace(".csv", ".png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
