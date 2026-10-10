import time
import csv
import os
from stable_baselines3 import PPO
from game.environment import ZombieSurvivalEnv, BomberCombatEnv
from utils.statistics import plot_metrics

def evaluate(agent_type="human", model_path="models/human/human_final_phase_3", episodes=10):
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
        env = BomberCombatEnv(config=config, render_mode="human")
    else:
        env = ZombieSurvivalEnv(config=config, render_mode="human")
    
    os.makedirs("logs", exist_ok=True)
    with open("logs/eval_metrics.csv", mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "steps", "human_survived", "zombies_killed", "outcome"])
        
        for ep in range(episodes):
            state, _ = env.reset()
            done = False
            step = 0
            
            init_zombies = sum(1 for z in env.game_engine.zombies)
            
            while not done:
                import random
                human_actions = {}
                bomber_actions = {}
                
                # Get the latest raw state
                raw_state = env.game_engine.get_state()
                
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
                if env.render_mode == "human":
                    env.renderer.draw_state(new_state)
                    
                done = new_state['outcome'] is not None
                step += 1
                time.sleep(0.016)
            
            outcome = env.game_engine.outcome
            survived = 1 if outcome == "victory" or sum(1 for h in env.game_engine.humans if h.is_alive) > 0 else 0
            final_zombies = sum(1 for z in env.game_engine.zombies if z.is_alive)
            killed = init_zombies - final_zombies
            
            writer.writerow([ep+1, step, survived, killed, outcome])
            print(f"Episode {ep + 1} finished in {step} steps. Outcome: {outcome}")
            
    # Plot metrics
    plot_metrics()

if __name__ == "__main__":
    evaluate("human")
