"""Gymnasium environments for separately trained human and bomber PPO policies."""
import warnings
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game.engine import GameEngine
from game.controllers import fallback_human_action, load_policy
from game.observations import OBS_DIM, observation, validate_capacity
from rl.rewards import human_reward_components, bomber_reward_components


class SurvivalEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}
    role = 'human'

    def __init__(self, render_mode=None, config=None, human_model_path=None, human_policy=None):
        super().__init__()
        if render_mode not in (None, 'human', 'rgb_array'):
            raise ValueError(f'Unsupported render_mode {render_mode}')
        self.game_engine = GameEngine(config)
        c = self.game_engine.config
        validate_capacity(c)
        if not c['num_humans'] or (self.role == 'bomber' and not c['num_bombers']):
            raise ValueError('Training requires a human and, for bomber training, a bomber.')
        self.action_space = spaces.Discrete(9 if self.role == 'human' else 10)
        self.observation_space = spaces.Box(-1, 1, (OBS_DIM,), dtype=np.float32)
        self.render_mode, self.renderer = render_mode, None
        self.human_policy = human_policy
        if human_model_path:
            self.human_policy = load_policy(human_model_path, 9)
        if self.role == 'bomber' and self.human_policy is None:
            warnings.warn('No human checkpoint supplied: using the deterministic survival controller.',
                          UserWarning, stacklevel=2)
        self.prev_state = None
        self.totals = {}

    def _get_obs_for_human(self, state, human):
        return observation(state, human, self.game_engine.config['max_steps'])

    def _get_obs_for_bomber(self, state, bomber):
        return observation(state, bomber, self.game_engine.config['max_steps'])

    def _get_obs(self, state=None):
        state = state if state is not None else self.game_engine.get_state()
        collection = state['humans' if self.role == 'human' else 'bombers']
        return observation(state, collection[0], self.game_engine.config['max_steps'])

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Gym owns the RNG; an unseeded reset continues its deterministic sequence.
        engine_seed = int(self.np_random.integers(0, 2**32))
        self.prev_state = self.game_engine.reset(seed=engine_seed)
        self.totals = dict(zombie_damage=0.0, zombie_kills=0, human_damage=0.0,
                           useful_explosions=0, wasted_explosions=0, episode_reward=0.0)
        self.component_totals = {}
        return self._get_obs(self.prev_state), {}

    def human_actions(self, state):
        actions = {}
        for h in state['humans']:
            if not h.is_alive or h.escaped:
                actions[h.id] = 0
            elif self.human_policy is not None:
                action, _ = self.human_policy.predict(self._get_obs_for_human(state, h), deterministic=True)
                actions[h.id] = int(action)
            else:
                actions[h.id] = fallback_human_action(self.game_engine, h)
        return actions

    def step(self, action):
        if self.prev_state is None:
            raise RuntimeError('Call reset() before step().')
        if self.prev_state['outcome'] is not None or self.prev_state['truncated']:
            raise RuntimeError('Episode ended; call reset() before stepping again.')
        if not self.action_space.contains(action):
            raise ValueError(f'Invalid action: {action}')
        if self.role == 'human':
            humans = {h.id: fallback_human_action(self.game_engine, h)
                      for h in self.prev_state['humans'] if h.id != 0}
            humans[0] = int(action)
            bombers = {}
        else:
            humans = self.human_actions(self.prev_state)
            bombers = {0: int(action)}
        state = self.game_engine.step(humans, bombers)
        # A human policy episode ends when its controlled human finishes, even if
        # other humans remain. Bomber episodes track the whole team's outcome.
        if self.role == 'human':
            h = state['humans'][0]
            if state['outcome'] is None and (h.escaped or not h.is_alive):
                state['outcome'] = 'victory' if h.escaped else 'defeat'
                state['truncated'] = False
        reward_fn = human_reward_components if self.role == 'human' else bomber_reward_components
        components = reward_fn(self.prev_state, state,
                               weights=self.game_engine.config.get(self.role+'_reward_weights'),
                               gamma=self.game_engine.config.get('gamma', 0.99))
        reward = float(sum(components.values()))
        self.prev_state = state
        self._record(state, components, reward)
        terminated, truncated = state['outcome'] is not None, state['truncated']
        info = {'reward_components': components, 'events': state['events']}
        if terminated or truncated:
            n = len(state['humans'])
            info['episode_stats'] = {**self.totals,
                'escape_rate': sum(h.escaped for h in state['humans'])/n,
                'death_rate': sum(not h.is_alive for h in state['humans'])/n,
                'survival_rate': sum(h.is_alive for h in state['humans'])/n,
                'steps': state['steps'],
                **{'reward_'+k: v for k, v in self.component_totals.items()}}
        if self.render_mode == 'human':
            self.render()
        return self._get_obs(state), reward, terminated, truncated, info

    def _record(self, state, components, reward):
        events = state['events']
        self.totals['episode_reward'] += reward
        self.totals['zombie_damage'] += sum(e.amount for e in events if e.kind == 'zombie_damage')
        self.totals['zombie_kills'] += sum(e.kind == 'zombie_killed' for e in events)
        self.totals['human_damage'] += sum(e.amount for e in events if e.kind == 'human_damage')
        useful = {e.actor_id for e in events if e.kind == 'zombie_damage' and e.amount > 0 and e.threatening}
        self.totals['useful_explosions'] += sum(e.kind == 'explosion' and e.actor_id in useful for e in events)
        self.totals['wasted_explosions'] += sum(e.kind == 'invalid_explosion' or
                                             (e.kind == 'explosion' and e.actor_id not in useful) for e in events)
        for key, value in components.items():
            self.component_totals[key] = self.component_totals.get(key, 0) + value

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from ui.renderer import Renderer
            self.renderer = Renderer(self.game_engine.config)
        self.renderer.draw_state(self.game_engine.get_state())
        if self.render_mode == 'rgb_array':
            import pygame
            return np.transpose(pygame.surfarray.array3d(self.renderer.screen), (1, 0, 2))
        return None

    def close(self):
        if self.renderer is not None:
            import pygame
            pygame.quit()
            self.renderer = None


class ZombieSurvivalEnv(SurvivalEnv):
    role = 'human'


class BomberCombatEnv(SurvivalEnv):
    role = 'bomber'
