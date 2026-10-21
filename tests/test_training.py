"""Small real PPO updates, checkpoint round trips and reproducible evaluation."""
import csv
import numpy as np
import pytest
import torch
from game.environment import BomberCombatEnv
from game.controllers import load_policy
from rl.train_human import train_human
from rl.train_bomber import train_bomber
from rl.evaluate_headless import evaluate_policies


def test_ppo_train_resume_frozen_human_and_comparison(tmp_path):
    config = dict(width=200, height=200, num_obstacles=0, num_safe_zones=1,
                  num_humans=1, num_zombies=1, max_steps=12)
    human_path = str(tmp_path/'human')
    human = train_human(timesteps=32, output=human_path, log_dir=str(tmp_path/'human_logs'),
                        seed=3, rollout_steps=16, config=config)
    assert human.num_timesteps == 32
    checkpoint = load_policy(human_path, 9)
    env = BomberCombatEnv(config=config, human_policy=checkpoint)
    original = {k: v.clone() for k, v in checkpoint.policy.state_dict().items()}
    env.reset(seed=7)
    env.step(0)
    assert all(torch.equal(original[k], v) for k, v in checkpoint.policy.state_dict().items())
    env.close()
    resumed = train_human(timesteps=16, output=human_path, log_dir=str(tmp_path/'resume_logs'),
                         resume=human_path, rollout_steps=16, config=config)
    assert resumed.num_timesteps == 48
    assert any(not torch.equal(original[k], v) for k, v in resumed.policy.state_dict().items())
    bomber_path = str(tmp_path/'bomber')
    train_bomber(timesteps=32, output=bomber_path, human_model_path=human_path,
                 log_dir=str(tmp_path/'bomber_logs'), seed=3, rollout_steps=16, config=config)
    with (tmp_path/'bomber_logs'/'progress.csv').open() as handle:
        logs = list(csv.DictReader(handle))
    for key in ('train/policy_gradient_loss', 'train/value_loss', 'train/loss',
                'episode/escape_rate', 'episode/zombie_damage', 'reward_step/invalid'):
        assert any(row.get(key) not in (None, '') for row in logs), key
    options = dict(mode='combined', human_model_path=human_path, bomber_model_path=bomber_path,
                   seeds=[4, 5], config=config)
    first = evaluate_policies(**options)
    second = evaluate_policies(**options)
    assert first == second
    baseline = evaluate_policies(**options, random_baseline=True)
    assert [r['seed'] for r in first] == [r['seed'] for r in baseline]
    assert all(np.isfinite(r['human_reward']) and np.isfinite(r['bomber_reward']) for r in first+baseline)


def test_old_observation_checkpoint_rejected(tmp_path):
    import gymnasium as gym
    from stable_baselines3 import PPO
    env = gym.make('CartPole-v1')
    model = PPO('MlpPolicy', env, n_steps=8, batch_size=8)
    path = str(tmp_path/'incompatible')
    model.save(path)
    with pytest.raises(ValueError, match='incompatible'):
        load_policy(path, 9)
    env.close()
