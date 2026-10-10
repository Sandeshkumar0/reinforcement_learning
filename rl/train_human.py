import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from game.environment import ZombieSurvivalEnv

def train_human():
    os.makedirs("models/human", exist_ok=True)
    os.makedirs("logs/human", exist_ok=True)

    # Curriculum phases
    phases = [
        {
            "name": "Phase 1: Navigation",
            "config": {'width': 800, 'height': 600, 'num_humans': 1, 'num_zombies': 0, 'num_bombers': 0, 'num_obstacles': 5, 'num_safe_zones': 1, 'max_steps': 500},
            "timesteps": 50000
        },
        {
            "name": "Phase 2: One Slow Zombie",
            "config": {'width': 800, 'height': 600, 'num_humans': 1, 'num_zombies': 1, 'num_bombers': 0, 'num_obstacles': 10, 'num_safe_zones': 1, 'max_steps': 1000},
            "timesteps": 100000
        },
        {
            "name": "Phase 3: Multiple Zombies",
            "config": {'width': 800, 'height': 600, 'num_humans': 1, 'num_zombies': 3, 'num_bombers': 0, 'num_obstacles': 15, 'num_safe_zones': 1, 'max_steps': 1000},
            "timesteps": 150000
        }
    ]

    model = None
    
    for i, phase in enumerate(phases):
        print(f"\\n--- Starting {phase['name']} ---")
        env = ZombieSurvivalEnv(config=phase["config"])
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=f'models/human/phase_{i+1}/',
            name_prefix='ppo_human'
        )

        if model is None:
            model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/human/")
        else:
            # Update the environment for the existing model
            model.set_env(env)
            
        model.learn(total_timesteps=phase["timesteps"], callback=checkpoint_callback, reset_num_timesteps=False)
        model.save(f"models/human/human_final_phase_{i+1}")
        
    print("Training Complete. Final model saved to models/human/human_final_phase_3")

if __name__ == "__main__":
    train_human()
