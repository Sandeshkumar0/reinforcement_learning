import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game.engine import GameEngine
from rl.rewards import compute_human_reward

class ZombieSurvivalEnv(gym.Env):
    """
    Gymnasium environment for training the Human survival agent.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        
        # 9 Discrete actions for human: 0=Stay, 1-8=Directions
        self.action_space = spaces.Discrete(9)
        
        # Observation space: 
        # [human_x, human_y, dist_to_safe, angle_to_safe, 
        #  z1_x, z1_y, z1_dist, z2_x, z2_y, z2_dist, z3_x, z3_y, z3_dist]
        # For simplicity, we assume up to 3 zombies. If fewer, pad with zeros.
        self.max_zombies = 3
        obs_dim = 4 + (3 * self.max_zombies)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.game_engine = GameEngine(config)
        self.render_mode = render_mode
        if self.render_mode == "human":
            from ui.renderer import Renderer
            self.renderer = Renderer(self.game_engine.config)
        
        self.prev_state = None

    def _get_obs_for_human(self, state, human):
        if not human:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        # Get safe zone center
        sz_x, sz_y = 0, 0
        if state['map'].safe_zones:
            sx, sy, sw, sh = state['map'].safe_zones[0]
            sz_x, sz_y = sx + sw/2, sy + sh/2
            
        dist_to_safe = np.hypot(human.x - sz_x, human.y - sz_y)
        angle_to_safe = np.arctan2(sz_y - human.y, sz_x - human.x)
        
        obs = [human.x, human.y, dist_to_safe, angle_to_safe]
        
        # Get up to max_zombies closest zombies
        zombies = state['zombies']
        zombies.sort(key=lambda z: human.distance_to(z) if z.is_alive else float('inf'))
        
        for i in range(self.max_zombies):
            if i < len(zombies) and zombies[i].is_alive:
                z = zombies[i]
                d = human.distance_to(z)
                obs.extend([z.x, z.y, d])
            else:
                obs.extend([-1, -1, -1]) # Padding
                
        return np.array(obs, dtype=np.float32)

    def _get_obs(self):
        state = self.game_engine.get_state()
        human = state['humans'][0] if state['humans'] else None
        return self._get_obs_for_human(state, human)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game_engine.reset()
        self.prev_state = state
        return self._get_obs(), {}

    def step(self, action):
        human_actions = {0: int(action)}
        
        # Bomber does nothing or random while human trains
        bomber_actions = {0: 0} 
        
        new_state = self.game_engine.step(human_actions, bomber_actions)
        
        obs = self._get_obs()
        reward = compute_human_reward(self.prev_state, new_state)
        
        terminated = new_state['outcome'] is not None
        truncated = False # handled by engine max_steps logic ending in outcome='draw'
        
        self.prev_state = new_state
        
        if self.render_mode == "human":
            self.renderer.draw_state(new_state)
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass


class BomberCombatEnv(gym.Env):
    """
    Gymnasium environment for training the Bomber combat agent.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, config=None):
        super().__init__()
        
        # 10 Discrete actions: 0=Stay, 1-8=Directions, 9=Explode
        self.action_space = spaces.Discrete(10)
        
        from rl.rewards import compute_bomber_reward
        self.compute_reward_fn = compute_bomber_reward
        
        # Observation space: 
        # [bomber_x, bomber_y, cooldown_ready, 
        #  z1_x, z1_y, z1_dist, z2_x, z2_y, z2_dist, ...] (max 5 zombies)
        self.max_zombies = 5
        obs_dim = 3 + (3 * self.max_zombies)
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.game_engine = GameEngine(config)
        self.render_mode = render_mode
        if self.render_mode == "human":
            from ui.renderer import Renderer
            self.renderer = Renderer(self.game_engine.config)
        
        self.prev_state = None

    def _get_obs_for_bomber(self, state, bomber):
        if not bomber:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
            
        obs = [bomber.x, bomber.y, 1.0 if bomber.cooldown <= 0 else 0.0]
        
        # Get up to max_zombies closest zombies
        zombies = [z for z in state['zombies'] if z.is_alive]
        zombies.sort(key=lambda z: bomber.distance_to(z))
        
        for i in range(self.max_zombies):
            if i < len(zombies):
                z = zombies[i]
                d = bomber.distance_to(z)
                obs.extend([z.x, z.y, d])
            else:
                obs.extend([-1, -1, -1]) # Padding
                
        return np.array(obs, dtype=np.float32)

    def _get_obs(self):
        state = self.game_engine.get_state()
        bomber = state['bombers'][0] if state['bombers'] else None
        return self._get_obs_for_bomber(state, bomber)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game_engine.reset()
        self.prev_state = state
        return self._get_obs(), {}

    def step(self, action):
        bomber_actions = {0: int(action)}
        
        # Human just moves randomly to not interfere with bomber training
        import random
        human_actions = {0: random.randint(0, 8)}
        
        new_state = self.game_engine.step(human_actions, bomber_actions)
        
        obs = self._get_obs()
        reward = self.compute_reward_fn(self.prev_state, new_state)
        
        terminated = new_state['outcome'] is not None
        truncated = False
        
        self.prev_state = new_state
        
        if self.render_mode == "human":
            self.renderer.draw_state(new_state)
            
        return obs, reward, terminated, truncated, {}

    def render(self):
        pass

    def close(self):
        pass

