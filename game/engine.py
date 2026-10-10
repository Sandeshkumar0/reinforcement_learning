from game.entities import Human, Zombie, Bomber
from game.map_generator import Map
from game.pathfinding import PathFinder
from game.combat import Explosion, apply_explosion_damage
import random
import math

class GameEngine:
    def __init__(self, config=None):
        if config is None:
            config = {
                'width': 800,
                'height': 600,
                'num_humans': 1,
                'num_zombies': 3,
                'num_bombers': 1,
                'num_obstacles': 15,
                'num_safe_zones': 1,
                'max_steps': 1000
            }
        self.config = config
        self.reset()

    def reset(self):
        self.map = Map(self.config['width'], self.config['height'])
        self.map.generate_random(self.config['num_obstacles'], self.config['num_safe_zones'])
        self.pathfinder = PathFinder(self.map, cell_size=20)
        
        self.humans = []
        self.zombies = []
        self.bombers = []
        self.explosions = []
        self.arrows = []
        self.steps = 0
        self.outcome = None # 'victory', 'defeat', 'draw'
        
        self._spawn_entities()
        return self.get_state()

    def _get_random_empty_pos(self, radius):
        while True:
            x = random.randint(radius, self.map.width - radius)
            y = random.randint(radius, self.map.height - radius)
            if not self.map.is_collision(x, y, radius):
                return x, y

    def _spawn_entities(self):
        for _ in range(self.config['num_humans']):
            x, y = self._get_random_empty_pos(5)
            self.humans.append(Human(x, y))
            
        for _ in range(self.config['num_zombies']):
            x, y = self._get_random_empty_pos(5)
            self.zombies.append(Zombie(x, y))
            
        for _ in range(self.config['num_bombers']):
            x, y = self._get_random_empty_pos(6)
            self.bombers.append(Bomber(x, y))

    def get_state(self):
        # A simplified state dictionary for RL or UI
        return {
            'humans': self.humans,
            'zombies': self.zombies,
            'bombers': self.bombers,
            'explosions': self.explosions,
            'arrows': self.arrows,
            'map': self.map,
            'steps': self.steps,
            'outcome': self.outcome
        }

    def step(self, human_actions=None, bomber_actions=None):
        if self.outcome is not None:
            return self.get_state()

        self.steps += 1
        
        # Move Humans
        if human_actions:
            # For simplicity, map discrete actions [0-8] to dx, dy
            # 0: stay, 1: N, 2: NE, 3: E, 4: SE, 5: S, 6: SW, 7: W, 8: NW
            action_map = {
                0: (0,0), 1: (0,-1), 2: (1,-1), 3: (1,0), 4: (1,1),
                5: (0,1), 6: (-1,1), 7: (-1,0), 8: (-1,-1)
            }
            for i, h in enumerate(self.humans):
                if h.is_alive and not h.escaped:
                    act = human_actions.get(i, 0)
                    dx, dy = action_map.get(act, (0,0))
                    new_x = h.x + dx * h.speed
                    new_y = h.y + dy * h.speed
                    if not self.map.is_collision(new_x, new_y, h.radius):
                        h.move(dx, dy)
                    
                    # Check safe zone
                    if self.map.in_safe_zone(h.x, h.y, h.radius):
                        h.escaped = True

        # Move Zombies (Rule-based A*)
        for z in self.zombies:
            if not z.is_alive:
                continue
                
            # Find nearest active human
            target = None
            min_dist = float('inf')
            for h in self.humans:
                if h.is_alive and not h.escaped:
                    d = z.distance_to(h)
                    if d < min_dist:
                        min_dist = d
                        target = h
            
            if target:
                waypoint = self.pathfinder.get_next_waypoint((z.x, z.y), (target.x, target.y))
                if waypoint:
                    dx = waypoint[0] - z.x
                    dy = waypoint[1] - z.y
                else:
                    # Direct fallback
                    dx = target.x - z.x
                    dy = target.y - z.y
                    
                new_x = z.x + (dx/math.hypot(dx,dy) if dx!=0 or dy!=0 else 0) * z.speed
                new_y = z.y + (dy/math.hypot(dx,dy) if dx!=0 or dy!=0 else 0) * z.speed
                
                if not self.map.is_collision(new_x, new_y, z.radius):
                    z.move(dx, dy)
                else:
                    # Slide along walls
                    if not self.map.is_collision(new_x, z.y, z.radius):
                        z.move(dx, 0)
                    elif not self.map.is_collision(z.x, new_y, z.radius):
                        z.move(0, dy)

            # Check human capture
            for h in self.humans:
                if h.is_alive and not h.escaped:
                    if z.distance_to(h) <= z.radius + h.radius:
                        h.is_alive = False

        # Move Bombers and Actions
        if bomber_actions:
            action_map = {
                0: (0,0), 1: (0,-1), 2: (1,-1), 3: (1,0), 4: (1,1),
                5: (0,1), 6: (-1,1), 7: (-1,0), 8: (-1,-1)
            }
            for i, b in enumerate(self.bombers):
                if b.is_alive:
                    b.update_cooldowns()
                    act = bomber_actions.get(i, 0) # 0-8 move, 9 explode
                    if act <= 8:
                        dx, dy = action_map.get(act, (0,0))
                        new_x = b.x + dx * b.speed
                        new_y = b.y + dy * b.speed
                        if not self.map.is_collision(new_x, new_y, b.radius):
                            b.move(dx, dy)
                    elif act == 9:
                        if b.trigger_explosion():
                            self.explosions.append(Explosion(b.x, b.y, b.explosion_radius))
                            
        # Update explosions
        active_exp = []
        for exp in self.explosions:
            exp.update()
            if exp.active:
                apply_explosion_damage(exp, self.zombies)
                active_exp.append(exp)
        self.explosions = active_exp

        self._check_outcome()
        return self.get_state()

    def _check_outcome(self):
        alive_humans = sum(1 for h in self.humans if h.is_alive and not h.escaped)
        escaped_humans = sum(1 for h in self.humans if h.escaped)
        total_humans_remaining = alive_humans + escaped_humans
        
        alive_zombies = sum(1 for z in self.zombies if z.is_alive)
        
        if alive_humans == 0:
            if escaped_humans > alive_zombies:
                self.outcome = 'victory'
            else:
                self.outcome = 'defeat'
        elif alive_zombies == 0:
            self.outcome = 'victory'
        elif self.steps >= self.config['max_steps']:
            if total_humans_remaining > alive_zombies:
                self.outcome = 'victory'
            elif alive_zombies > total_humans_remaining:
                self.outcome = 'defeat'
            else:
                self.outcome = 'draw'
