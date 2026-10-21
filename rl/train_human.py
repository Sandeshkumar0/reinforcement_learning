from rl.training import train_phases


def train_human(timesteps=300000, output='models/human/human_final_phase_3',
                log_dir='logs/human', seed=0, resume=None, rollout_steps=1024, config=None):
    if timesteps < 3:
        raise ValueError('Human training requires at least 3 requested timesteps.')
    base = dict(num_humans=1, num_bombers=0, num_safe_zones=1, end_on_zombie_clear=False)
    if resume or config is not None:
        phases = [dict(name='Survival', config={**base, **(config or {})}, timesteps=timesteps)]
    else:
        amounts = [timesteps//6, timesteps//3]
        amounts.append(timesteps-sum(amounts))
        phases = [
            dict(name='Navigation', config={**base, 'num_zombies': 0, 'num_obstacles': 5}, timesteps=max(1, amounts[0])),
            dict(name='One slow zombie', config={**base, 'num_zombies': 1, 'zombie_speed': 1.0, 'num_obstacles': 10}, timesteps=max(1, amounts[1])),
            dict(name='Survival', config={**base, 'num_zombies': 3, 'num_obstacles': 15}, timesteps=max(1, amounts[2]))]
    return train_phases('human', phases, output, log_dir, seed, resume, rollout_steps=rollout_steps)


if __name__ == '__main__':
    train_human()
