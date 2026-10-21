"""Shared PPO training, checkpoint compatibility and episode metric logging."""
import importlib.util
import json
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from game.controllers import load_policy
from game.observations import OBS_DIM


class EpisodeMetrics(BaseCallback):
    def _on_step(self):
        for info in self.locals['infos']:
            for key, value in info.get('reward_components', {}).items():
                self.logger.record_mean('reward_step/'+key, value)
            for key, value in info.get('episode_stats', {}).items():
                self.logger.record_mean('episode/'+key, value)
        return True


def train_phases(role, phases, output, log_dir, seed=0, resume=None,
                 human_model_path=None, rollout_steps=1024):
    from game.environment import ZombieSurvivalEnv, BomberCombatEnv
    if rollout_steps < 2:
        raise ValueError('rollout_steps must be at least 2')
    torch.set_num_threads(1)
    output, log_dir = Path(output), Path(log_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    formats = ['stdout', 'csv']
    if importlib.util.find_spec('tensorboard'):
        formats.append('tensorboard')
    logger = configure(str(log_dir), formats)
    model = load_policy(resume, 9 if role == 'human' else 10) if resume else None
    final_config = None
    for phase_index, phase in enumerate(phases):
        env_cls = ZombieSurvivalEnv if role == 'human' else BomberCombatEnv
        env = env_cls(config=phase['config'], human_model_path=human_model_path)
        final_config = env.game_engine.config
        env.reset(seed=seed+phase_index)
        monitored = Monitor(env)
        try:
            if model is None:
                batch_size = min(64, rollout_steps)
                while rollout_steps % batch_size:
                    batch_size -= 1
                model = PPO('MlpPolicy', monitored, seed=seed, device='cpu', verbose=1,
                            n_steps=rollout_steps, batch_size=batch_size,
                            gamma=final_config.get('gamma', 0.99))
            else:
                if model.gamma != final_config.get('gamma', 0.99):
                    raise ValueError('Resume gamma must match reward shaping gamma.')
                model.set_env(monitored)
            model.set_logger(logger)
            callbacks = CallbackList([
                EpisodeMetrics(),
                CheckpointCallback(save_freq=max(rollout_steps, 10000),
                                   save_path=str(output.parent/'checkpoints'), name_prefix='ppo_'+role)
            ])
            print(f"Training {role}: {phase['name']} ({phase['timesteps']} requested steps)")
            model.learn(total_timesteps=phase['timesteps'], callback=callbacks, reset_num_timesteps=False)
            model.save(str(output))
            # Persist both config and observation version alongside each final checkpoint.
            output.with_suffix('.json').write_text(json.dumps(dict(schema_version=2, observation_dim=OBS_DIM,
                role=role, config=final_config, human_model_path=human_model_path, seed=seed,
                num_timesteps=model.num_timesteps), indent=2))
        finally:
            monitored.close()
    # PPO logs optimization metrics after collecting a rollout. Flush the last update too.
    logger.dump(model.num_timesteps)
    logger.close()
    return model
