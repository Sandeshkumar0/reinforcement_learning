"""Reproducible short integration run; not a policy-quality benchmark.

Run: python -m rl.verify_training --timesteps 1024 --episodes 10
"""
import argparse
import hashlib
import json
from pathlib import Path
from rl.train_human import train_human
from rl.train_bomber import train_bomber
from rl.evaluate_headless import evaluate_policies, write_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--timesteps', type=int, default=1024)
    parser.add_argument('--episodes', type=int, default=10)
    args = parser.parse_args()
    if args.timesteps < 3 or args.episodes < 1:
        parser.error('Require at least 3 timesteps and 1 episode.')
    root = Path('logs/smoke_v2')
    root.mkdir(parents=True, exist_ok=True)
    config = dict(width=240, height=240, num_humans=1, num_zombies=2,
                  num_bombers=1, num_obstacles=0, num_safe_zones=1, max_steps=160,
                  explosion_damage=60, explosion_cooldown=30, zombie_attack_cooldown=20)
    (root/'config.json').write_text(json.dumps(config, indent=2))
    human_path, bomber_path = 'models/smoke_v2/human', 'models/smoke_v2/bomber'
    human = train_human(timesteps=args.timesteps, output=human_path,
                        log_dir=str(root/'human'), seed=7, rollout_steps=128,
                        config={**config, 'num_bombers': 0, 'end_on_zombie_clear': False})
    digest = hashlib.sha256(Path(human_path+'.zip').read_bytes()).hexdigest()
    bomber = train_bomber(timesteps=args.timesteps, output=bomber_path,
                          log_dir=str(root/'bomber'), seed=8, rollout_steps=128,
                          config=config, human_model_path=human_path)
    assert digest == hashlib.sha256(Path(human_path+'.zip').read_bytes()).hexdigest()
    seeds = list(range(100, 100+args.episodes))
    options = dict(mode='combined', human_model_path=human_path,
                   bomber_model_path=bomber_path, seeds=seeds, config=config)
    trained = evaluate_policies(**options)
    random = evaluate_policies(**options, random_baseline=True)
    write_results(trained+random, root/'comparison.csv')
    metrics = ['escape_rate', 'death_rate', 'zombie_damage', 'zombies_killed',
               'useful_explosions', 'wasted_explosions', 'human_reward', 'bomber_reward']
    report = dict(purpose='Integration smoke test only; no claim of policy improvement.',
                  human_training_steps=human.num_timesteps, bomber_training_steps=bomber.num_timesteps,
                  frozen_human_checkpoint_unchanged=True, evaluation_seeds=seeds,
                  configuration=config,
                  mean_metrics={label: {k: sum(r[k] for r in rows)/len(rows) for k in metrics}
                                for label, rows in [('trained', trained), ('random', random)]})
    (root/'report.json').write_text(json.dumps(report, indent=2))
    print('Verification report:', root/'report.json')


if __name__ == '__main__':
    main()
