"""Train, resume, evaluate or compare the human and bomber policies."""
import argparse
import json
from pathlib import Path


def run_random_play():
    import random
    import pygame
    from game.engine import GameEngine
    from ui.renderer import Renderer
    engine = GameEngine()
    renderer = Renderer(engine.config)
    try:
        while renderer.process_events():
            state = engine.step({h.id: random.randrange(9) for h in engine.humans},
                                {b.id: random.randrange(10) for b in engine.bombers})
            renderer.draw_state(state)
            if state['outcome'] is not None or state['truncated']:
                engine.reset()
    finally:
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('play', 'train-human', 'train-bomber', 'train-all',
                 'evaluate-human', 'evaluate-bomber', 'evaluate-combined'):
        modes.add_argument('--'+mode, action='store_true')
    parser.add_argument('--human-model', default='models/human/human_final_phase_3')
    parser.add_argument('--bomber-model', default='models/bomber/bomber_final')
    parser.add_argument('--timesteps', type=int, help='Requested steps per agent (PPO rounds to full rollouts).')
    parser.add_argument('--rollout-steps', type=int, default=1024)
    parser.add_argument('--resume', help='Compatible checkpoint; used only with a single training mode.')
    parser.add_argument('--config', help='JSON object overriding engine config and reward weights.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--compare', action='store_true', help='Also evaluate random actions on identical seeds.')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output', help='Evaluation CSV output path.')
    args = parser.parse_args()
    if args.resume and not (args.train_human or args.train_bomber):
        parser.error('--resume requires --train-human or --train-bomber')
    config = json.loads(Path(args.config).read_text()) if args.config else None
    if args.resume and config is None:
        from rl.evaluate_headless import checkpoint_config
        config = checkpoint_config(args.resume) or None
    if args.play:
        run_random_play()
        return
    if args.train_human or args.train_all:
        from rl.train_human import train_human
        train_human(timesteps=args.timesteps if args.timesteps is not None else 300000,
                    output=args.human_model, seed=args.seed, resume=args.resume,
                    rollout_steps=args.rollout_steps, config=config)
    if args.train_bomber or args.train_all:
        from rl.train_bomber import train_bomber
        hp = args.human_model
        if not Path(hp).is_file() and not Path(hp+'.zip').is_file():
            if hp != 'models/human/human_final_phase_3':
                parser.error(f'Human checkpoint does not exist: {hp}')
            hp = None
        train_bomber(timesteps=args.timesteps if args.timesteps is not None else 200000,
                     human_model_path=hp, output=args.bomber_model, seed=args.seed,
                     resume=args.resume, rollout_steps=args.rollout_steps, config=config)
    if args.evaluate_human or args.evaluate_bomber or args.evaluate_combined:
        from rl.evaluate_headless import evaluate_policies, write_results
        if args.episodes < 1:
            parser.error('--episodes must be positive')
        mode = 'human' if args.evaluate_human else 'bomber' if args.evaluate_bomber else 'combined'
        hp = args.human_model
        if mode == 'bomber' and hp == 'models/human/human_final_phase_3' and not Path(hp+'.zip').exists():
            hp = None
        kwargs = dict(mode=mode, human_model_path=hp, bomber_model_path=args.bomber_model,
                      seeds=range(args.seed, args.seed+args.episodes), config=config)
        rows = evaluate_policies(**kwargs, render=args.render)
        if args.compare:
            rows += evaluate_policies(**kwargs, random_baseline=True)
        write_results(rows, args.output or f'logs/eval_{mode}_metrics.csv')


if __name__ == '__main__':
    main()
