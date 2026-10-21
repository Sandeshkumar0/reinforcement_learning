"""Plot evaluation CSVs, including fixed-seed trained/random comparisons."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def plot_metrics(csv_path='logs/eval_metrics.csv', output_path='logs/metrics_plot.png'):
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    survival = next((k for k in ('escape_rate', 'humans_survived', 'human_survived') if k in df), None)
    if survival is None:
        raise ValueError('CSV has no human survival or escape metric.')
    metrics = [(survival, survival.replace('_', ' ').title()), ('steps', 'Steps per Episode'),
               ('zombies_killed', 'Zombies Killed')]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    groups = df.groupby('controller') if 'controller' in df else [('policy', df)]
    for label, group in groups:
        x = group['seed'] if 'seed' in group else group['episode']
        for ax, (column, title) in zip(axes, metrics):
            ax.plot(x, group[column], marker='o', label=label)
            ax.set_title(title)
            ax.set_xlabel('Seed' if 'seed' in group else 'Episode')
            ax.legend()
    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    print(f'Plot saved to {output}')
