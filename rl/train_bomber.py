from pathlib import Path
from rl.training import train_phases


def train_bomber(timesteps=200000, human_model_path=None, output='models/bomber/bomber_final',
                 log_dir='logs/bomber', seed=0, resume=None, rollout_steps=1024, config=None):
    if timesteps < 1:
        raise ValueError('timesteps must be positive')
    if human_model_path is None:
        default = Path('models/human/human_final_phase_3.zip')
        if default.exists():
            human_model_path = str(default)
    phase = dict(name='Protect humans', config={
        'num_humans': 1, 'num_zombies': 5, 'num_bombers': 1, 'num_obstacles': 10,
        **(config or {})}, timesteps=timesteps)
    return train_phases('bomber', [phase], output, log_dir, seed, resume,
                        human_model_path, rollout_steps)


if __name__ == '__main__':
    train_bomber()
