import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from game.environment import BomberCombatEnv

def train_bomber():
    os.makedirs("models/bomber", exist_ok=True)
    os.makedirs("logs/bomber", exist_ok=True)

    config = {
        'width': 800, 'height': 600, 
        'num_humans': 1, 'num_zombies': 5, 'num_bombers': 1, 
        'num_obstacles': 10, 'num_safe_zones': 1, 'max_steps': 1000
    }
    
    print("--- Starting Bomber Training ---")
    env = BomberCombatEnv(config=config)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=20000, 
        save_path='models/bomber/checkpoints/',
        name_prefix='ppo_bomber'
    )

    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="logs/bomber/")
    # Train for 200k steps to learn combat strategy
    model.learn(total_timesteps=200000, callback=checkpoint_callback)
    model.save("models/bomber/bomber_final")
    print("Training Complete. Final model saved to models/bomber/bomber_final")

if __name__ == "__main__":
    train_bomber()
