"""Rendered evaluation compatibility entry point."""
from rl.evaluate_headless import evaluate_policies, write_results


def evaluate(agent_type='human', model_path='models/human/human_final_phase_3', episodes=10):
    key = 'human_model_path' if agent_type == 'human' else 'bomber_model_path'
    rows = evaluate_policies(mode=agent_type, seeds=range(episodes), render=True, **{key: model_path})
    write_results(rows, f'logs/eval_{agent_type}_metrics.csv')
    return rows
