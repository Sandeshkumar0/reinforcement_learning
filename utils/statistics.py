import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_metrics(csv_path="logs/eval_metrics.csv"):
    if not os.path.exists(csv_path):
        print(f"Metrics file {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Human Survival Rate
    plt.subplot(1, 3, 1)
    plt.plot(df['episode'], df['human_survived'], label="Human Survived", marker='o')
    plt.title("Human Survival vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Survived (1=Yes, 0=No)")
    
    # Subplot 2: Steps to Win
    plt.subplot(1, 3, 2)
    plt.plot(df['episode'], df['steps'], label="Steps", color='orange', marker='o')
    plt.title("Steps per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    
    # Subplot 3: Zombies Eliminated
    plt.subplot(1, 3, 3)
    plt.plot(df['episode'], df['zombies_killed'], label="Killed", color='red', marker='o')
    plt.title("Zombies Eliminated")
    plt.xlabel("Episode")
    plt.ylabel("Count")
    
    plt.tight_layout()
    plt.savefig("logs/metrics_plot.png")
    print("Plot saved to logs/metrics_plot.png")
