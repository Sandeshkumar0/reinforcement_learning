import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("logs", exist_ok=True)

# --- Human Agent Metrics ---
human_df = pd.read_csv("logs/eval_human_metrics.csv")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Human Agent Evaluation (1000×800 map, 5 Humans vs 5 Zombies)", fontsize=13, fontweight='bold')

axes[0].bar(human_df['episode'], human_df['steps'], color='steelblue', edgecolor='black')
axes[0].axhline(y=human_df['steps'].mean(), color='red', linestyle='--', label=f"Avg: {human_df['steps'].mean():.0f}")
axes[0].set_title("Steps Survived per Episode")
axes[0].set_xlabel("Episode")
axes[0].set_ylabel("Steps")
axes[0].legend()

axes[1].bar(human_df['episode'], human_df['humans_survived'], color='green', edgecolor='black')
axes[1].axhline(y=human_df['humans_survived'].mean(), color='red', linestyle='--', label=f"Avg: {human_df['humans_survived'].mean():.1f}")
axes[1].set_title("Humans Still Alive at End")
axes[1].set_xlabel("Episode")
axes[1].set_ylabel("Count (out of 5)")
axes[1].set_ylim(0, 5)
axes[1].legend()

axes[2].bar(human_df['episode'], human_df['zombies_killed'], color='orange', edgecolor='black')
axes[2].set_title("Zombies Killed")
axes[2].set_xlabel("Episode")
axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("logs/human_eval_v2_plot.png", dpi=120)
print("Saved: logs/human_eval_v2_plot.png")

# --- Bomber Agent Metrics ---
bomber_df = pd.read_csv("logs/eval_bomber_metrics.csv")

fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
fig2.suptitle("Bomber Agent Evaluation (1000×800 map, 5 Humans vs 5 Zombies + Bomber)", fontsize=13, fontweight='bold')

axes2[0].bar(bomber_df['episode'], bomber_df['steps'], color='darkorange', edgecolor='black')
axes2[0].axhline(y=bomber_df['steps'].mean(), color='red', linestyle='--', label=f"Avg: {bomber_df['steps'].mean():.0f}")
axes2[0].set_title("Steps per Episode")
axes2[0].set_xlabel("Episode")
axes2[0].set_ylabel("Steps")
axes2[0].legend()

axes2[1].bar(bomber_df['episode'], bomber_df['zombies_killed'], color='crimson', edgecolor='black')
axes2[1].set_title("Zombies Killed by Bomber")
axes2[1].set_xlabel("Episode")
axes2[1].set_ylabel("Count")

plt.tight_layout()
plt.savefig("logs/bomber_eval_v2_plot.png", dpi=120)
print("Saved: logs/bomber_eval_v2_plot.png")

# Print summary
print(f"\n=== HUMAN AGENT SUMMARY ===")
print(f"Avg steps:          {human_df['steps'].mean():.1f}")
print(f"Max steps:          {human_df['steps'].max()}")
print(f"Avg humans alive:   {human_df['humans_survived'].mean():.2f}/5")
print(f"Best episode:       Ep{human_df.loc[human_df['steps'].idxmax(), 'episode']} ({human_df['steps'].max()} steps)")

print(f"\n=== BOMBER AGENT SUMMARY ===")
print(f"Avg steps:          {bomber_df['steps'].mean():.1f}")
print(f"Max steps:          {bomber_df['steps'].max()}")
print(f"Total zombies killed: {bomber_df['zombies_killed'].sum()}")
