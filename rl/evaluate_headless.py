"""
Headless evaluation script - no PyGame rendering.
Runs episodes and outputs metrics to CSV and stdout.
"""
import csv
import os
import sys
import random
from stable_baselines3 import PPO
from game.environment import ZombieSurvivalEnv, BomberCombatEnv

def evaluate_headless(agent_type="human", model_path="models/human/human_final_phase_3", episodes=10):
    print(f"Loading {agent_type} model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    config = {
        'width': 1000, 'height': 800,
        'num_humans': 5, 'num_zombies': 5, 'num_bombers': 1 if agent_type == "bomber" else 0,
        'num_obstacles': 15, 'num_safe_zones': 1, 'max_steps': 1000
    }

    if agent_type == "bomber":
        env = BomberCombatEnv(config=config, render_mode=None)
    else:
        env = ZombieSurvivalEnv(config=config, render_mode=None)

    os.makedirs("logs", exist_ok=True)
    out_file = f"logs/eval_{agent_type}_metrics.csv"
    with open(out_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "humans_survived", "zombies_killed", "outcome"])

        for ep in range(episodes):
            env.reset()
            done = False
            step = 0
            init_zombies = len(env.game_engine.zombies)

            while not done:
                raw_state = env.game_engine.get_state()
                human_actions = {}
                bomber_actions = {}

                if agent_type == "human":
                    for i, h in enumerate(env.game_engine.humans):
                        if h.is_alive and not h.escaped:
                            obs = env._get_obs_for_human(raw_state, h)
                            action, _ = model.predict(obs, deterministic=True)
                            human_actions[i] = int(action)
                    for i in range(len(env.game_engine.bombers)):
                        bomber_actions[i] = random.randint(0, 8)
                else:
                    for i in range(len(env.game_engine.humans)):
                        human_actions[i] = random.randint(0, 8)
                    for i, b in enumerate(env.game_engine.bombers):
                        if b.is_alive:
                            obs = env._get_obs_for_bomber(raw_state, b)
                            action, _ = model.predict(obs, deterministic=True)
                            bomber_actions[i] = int(action)

                new_state = env.game_engine.step(human_actions, bomber_actions)
                done = new_state['outcome'] is not None
                step += 1

            outcome = env.game_engine.outcome
            survived = sum(1 for h in env.game_engine.humans if h.is_alive or h.escaped)
            final_zombies = sum(1 for z in env.game_engine.zombies if z.is_alive)
            killed = init_zombies - final_zombies

            writer.writerow([ep+1, step, survived, killed, outcome])
            print(f"Episode {ep+1:2d}: {step:5d} steps | {survived}/5 humans alive | {killed} zombies killed | {outcome}")

    print(f"\nMetrics saved to {out_file}")

if __name__ == "__main__":
    agent = sys.argv[1] if len(sys.argv) > 1 else "human"
    model = sys.argv[2] if len(sys.argv) > 2 else f"models/{agent}/{'human_final_phase_3' if agent == 'human' else 'bomber_final'}"
    evaluate_headless(agent, model)
