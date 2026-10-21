"""Fixed-seed policy evaluation. Never calls learn() or changes model weights."""
import csv
import json
import random
from pathlib import Path
from game.engine import GameEngine
from game.controllers import load_policy, fallback_human_action
from game.observations import observation
from rl.rewards import human_reward_components, bomber_reward_components


def checkpoint_config(path):
    if path:
        metadata = Path(path).with_suffix('.json')
        if metadata.exists():
            return json.loads(metadata.read_text())['config']
    return {}


def evaluate_policies(mode='combined', human_model_path=None, bomber_model_path=None,
                      seeds=range(10), random_baseline=False, config=None, render=False):
    if mode not in ('human', 'bomber', 'combined'):
        raise ValueError('mode must be human, bomber or combined')
    human = load_policy(human_model_path, 9) if human_model_path and (not random_baseline or mode == 'bomber') else None
    bomber = load_policy(bomber_model_path, 10) if bomber_model_path and not random_baseline and mode != 'human' else None
    if not random_baseline and mode in ('human', 'combined') and human is None:
        raise ValueError('A trained human checkpoint is required for this evaluation mode.')
    if not random_baseline and mode in ('bomber', 'combined') and bomber is None:
        raise ValueError('A trained bomber checkpoint is required for this evaluation mode.')
    saved_config = checkpoint_config(human_model_path if mode == 'human' else bomber_model_path)
    c = {**saved_config, **(config or {})}
    if mode == 'human':
        c['num_bombers'] = 0
        c['end_on_zombie_clear'] = False
    engine = GameEngine(c)
    from game.observations import validate_capacity
    validate_capacity(engine.config)
    renderer = None
    if render:
        from ui.renderer import Renderer
        renderer = Renderer(engine.config)
    rows = []
    try:
        for episode, seed in enumerate(seeds, 1):
            state = engine.reset(seed=int(seed))
            rng = random.Random(int(seed)+100000)
            row = dict(episode=episode, seed=int(seed), controller='random' if random_baseline else 'trained',
                       human_reward=0.0, bomber_reward=0.0, zombie_damage=0.0, zombies_killed=0,
                       human_damage=0.0, useful_explosions=0, wasted_explosions=0)
            while state['outcome'] is None and not state['truncated']:
                ha, ba = {}, {}
                for h in state['humans']:
                    if not h.is_alive or h.escaped:
                        continue
                    if human:
                        action, _ = human.predict(observation(state, h, engine.config['max_steps']), deterministic=True)
                        ha[h.id] = int(action)
                    elif mode == 'bomber':
                        # Same frozen/fallback humans for trained and random bomber comparisons.
                        ha[h.id] = fallback_human_action(engine, h)
                    else:
                        ha[h.id] = rng.randrange(9)
                for b in state['bombers']:
                    if bomber:
                        action, _ = bomber.predict(observation(state, b, engine.config['max_steps']), deterministic=True)
                        ba[b.id] = int(action)
                    else:
                        ba[b.id] = rng.randrange(10)
                new_state = engine.step(ha, ba)
                for h in state['humans']:
                    row['human_reward'] += sum(human_reward_components(state, new_state, h.id,
                        weights=engine.config.get('human_reward_weights'), gamma=engine.config.get('gamma', 0.99)).values())
                for b in state['bombers']:
                    row['bomber_reward'] += sum(bomber_reward_components(state, new_state, b.id,
                        weights=engine.config.get('bomber_reward_weights'), gamma=engine.config.get('gamma', 0.99)).values())
                events = new_state['events']
                row['zombie_damage'] += sum(e.amount for e in events if e.kind == 'zombie_damage')
                row['human_damage'] += sum(e.amount for e in events if e.kind == 'human_damage')
                row['zombies_killed'] += sum(e.kind == 'zombie_killed' for e in events)
                useful = {e.actor_id for e in events if e.kind == 'zombie_damage' and e.threatening}
                row['useful_explosions'] += sum(e.kind == 'explosion' and e.actor_id in useful for e in events)
                row['wasted_explosions'] += sum(e.kind == 'invalid_explosion' or (e.kind == 'explosion' and e.actor_id not in useful) for e in events)
                state = new_state
                if renderer:
                    if not renderer.process_events():
                        return rows
                    renderer.draw_state(state)
            n = len(state['humans'])
            row.update(steps=state['steps'], outcome=state['outcome'] or 'timeout',
                       humans_survived=sum(h.is_alive for h in state['humans']),
                       escape_rate=sum(h.escaped for h in state['humans'])/n,
                       death_rate=sum(not h.is_alive for h in state['humans'])/n)
            rows.append(row)
    finally:
        if renderer:
            import pygame
            pygame.quit()
    return rows


def write_results(rows, output):
    if not rows:
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {len(rows)} episodes to {path}')
    for controller in sorted({r['controller'] for r in rows}):
        group = [r for r in rows if r['controller'] == controller]
        averages = {key: round(sum(r[key] for r in group)/len(group), 3)
                    for key in ('escape_rate', 'death_rate', 'zombie_damage', 'zombies_killed',
                                'useful_explosions', 'wasted_explosions', 'human_reward', 'bomber_reward')}
        print(controller, json.dumps(averages))


def evaluate_headless(agent_type='human', model_path='models/human/human_final_phase_3', episodes=10):
    kwargs = {'human_model_path' if agent_type == 'human' else 'bomber_model_path': model_path}
    rows = evaluate_policies(mode=agent_type, seeds=range(episodes), **kwargs)
    write_results(rows, f'logs/eval_{agent_type}_metrics.csv')
    return rows


if __name__ == '__main__':
    import sys
    agent = sys.argv[1] if len(sys.argv) > 1 else 'human'
    path = sys.argv[2] if len(sys.argv) > 2 else f'models/{agent}/'+('human_final_phase_3' if agent == 'human' else 'bomber_final')
    evaluate_headless(agent, path)
