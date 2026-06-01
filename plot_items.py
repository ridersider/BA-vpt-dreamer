import pandas as pd
import matplotlib.pyplot as plt
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "logs/run_1779963538/item_log.csv"
df = pd.read_csv(path)

# Cumulative sum of delta per item (ignores resets in 'total' column)
totals = df.groupby("item_id")["delta"].sum()
items = totals.sort_values(ascending=False).index
fig, ax = plt.subplots(figsize=(14, 6))

for item in items:
    sub = df[df["item_id"] == item].sort_values("step")
    cum = sub["delta"].cumsum()
    ax.plot(sub["step"].values, cum.values, label=f"{item} ({totals[item]})", linewidth=1.5)

ax.set_xlabel("Step")
ax.set_ylabel("Cumulative items collected")
ax.set_title(f"Item collection over time — {path}")
ax.legend(loc="upper left", fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)
plt.tight_layout()

out = path.replace(".csv", "_cumulative.png")
plt.savefig(out, dpi=150)
print(f"Saved: {out}")
plt.show()
