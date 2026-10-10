from stable_baselines3.common.env_checker import check_env
from game.environment import ZombieSurvivalEnv

def test_environment():
    env = ZombieSurvivalEnv()
    check_env(env, warn=True)
    print("Environment check passed!")

if __name__ == "__main__":
    test_environment()
