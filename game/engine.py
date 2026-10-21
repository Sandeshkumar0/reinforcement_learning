"""Deterministic combat order: cooldowns, humans, zombies, bombers, attacks."""
import copy
import math
import random
from game.entities import Human, Zombie, Bomber
from game.map_generator import Map
from game.pathfinding import PathFinder
from game.combat import CombatEvent, Explosion, apply_explosion_damage

ACTION_MAP = {0: (0, 0), 1: (0, -1), 2: (1, -1), 3: (1, 0), 4: (1, 1),
              5: (0, 1), 6: (-1, 1), 7: (-1, 0), 8: (-1, -1)}
DEFAULT_CONFIG = dict(width=800, height=600, num_humans=1, num_zombies=3,
                      num_bombers=1, num_obstacles=15, num_safe_zones=1, max_steps=1000,
                      human_health=100, zombie_health=100, human_speed=2.5,
                      zombie_speed=1.5, bomber_speed=2.0, zombie_attack_damage=25,
                      zombie_attack_range=12, zombie_attack_cooldown=30,
                      threat_distance=100, explosion_radius=70, explosion_damage=100,
                      explosion_cooldown=60, end_on_zombie_clear=True)


class GameEngine:
    def __init__(self, config=None):
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        c = self.config
        if c['width'] < 100 or c['height'] < 100:
            raise ValueError('Map dimensions must be at least 100.')
        for key in ('num_humans', 'num_zombies', 'num_bombers', 'num_obstacles', 'num_safe_zones'):
            if not isinstance(c[key], int) or c[key] < 0:
                raise ValueError(f'{key} must be a nonnegative integer')
        for key in ('human_health', 'zombie_health', 'zombie_attack_range', 'zombie_attack_damage',
                    'explosion_radius', 'explosion_damage', 'max_steps',
                    'zombie_attack_cooldown', 'explosion_cooldown'):
            if c[key] <= 0:
                raise ValueError(f'{key} must be positive')
        if c['threat_distance'] < c['zombie_attack_range']:
            raise ValueError('threat_distance must cover zombie_attack_range')
        self.rng = random.Random()
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(int(seed))
        c = self.config
        self.map = Map(c['width'], c['height'])
        self.map.generate_random(c['num_obstacles'], c['num_safe_zones'], rng=self.rng)
        self.pathfinder = PathFinder(self.map, cell_size=20)
        self.humans, self.zombies, self.bombers = [], [], []
        self.explosions, self.arrows, self.events = [], [], []
        self.steps, self.outcome, self.truncated = 0, None, False
        self._spawn_entities()
        self._update_targets()
        return self.get_state()

    def _get_random_empty_pos(self, radius, occupied):
        for _ in range(10000):
            x = self.rng.uniform(radius, self.map.width - radius)
            y = self.rng.uniform(radius, self.map.height - radius)
            if (not self.map.is_collision(x, y, radius)
                    and not self.map.in_safe_zone(x, y, radius)
                    and all(math.hypot(x-e.x, y-e.y) > radius+e.radius+2 for e in occupied)):
                return x, y
        raise ValueError('Cannot spawn entities: reduce obstacles/entities or increase map size.')

    def _spawn_entities(self):
        c, occupied = self.config, []
        for i in range(c['num_humans']):
            h = Human(*self._get_random_empty_pos(5, occupied), entity_id=i,
                      health=c['human_health'], speed=c['human_speed'])
            self.humans.append(h)
            occupied.append(h)
        for i in range(c['num_zombies']):
            z = Zombie(*self._get_random_empty_pos(5, occupied), entity_id=i,
                       health=c['zombie_health'], speed=c['zombie_speed'],
                       attack_damage=c['zombie_attack_damage'], attack_range=c['zombie_attack_range'],
                       attack_cooldown=c['zombie_attack_cooldown'])
            self.zombies.append(z)
            occupied.append(z)
        for i in range(c['num_bombers']):
            b = Bomber(*self._get_random_empty_pos(6, occupied), entity_id=i, speed=c['bomber_speed'],
                       explosion_radius=c['explosion_radius'], explosion_damage=c['explosion_damage'],
                       explosion_cooldown=c['explosion_cooldown'])
            self.bombers.append(b)
            occupied.append(b)

    def get_state(self):
        # Detached snapshots: UI/observation consumers cannot mutate the world or prior states.
        return copy.deepcopy(dict(humans=self.humans, zombies=self.zombies, bombers=self.bombers,
                                  explosions=self.explosions, arrows=self.arrows, map=self.map,
                                  steps=self.steps, outcome=self.outcome, truncated=self.truncated,
                                  events=tuple(self.events)))

    def _update_targets(self):
        active = [h for h in self.humans if h.is_alive and not h.escaped]
        for z in self.zombies:
            target = min(active, key=lambda h: (z.distance_to(h), h.id)) if active and z.is_alive else None
            z.target = target.id if target else None
            z.is_threatening = bool(target and z.distance_to(target) <= self.config['threat_distance'])

    def _move(self, entity, dx, dy, slide=False):
        length = math.hypot(dx, dy)
        if not length:
            return
        dx, dy = dx / length * entity.speed, dy / length * entity.speed
        for mx, my in ([(dx, dy), (dx, 0), (0, dy)] if slide else [(dx, dy)]):
            if not self.map.is_collision(entity.x + mx, entity.y + my, entity.radius):
                entity.x += mx
                entity.y += my
                return

    def step(self, human_actions=None, bomber_actions=None):
        if self.outcome is not None or self.truncated:
            # No repeated terminal event rewards if a caller mistakenly steps again.
            self.events = []
            return self.get_state()
        human_actions, bomber_actions = human_actions or {}, bomber_actions or {}
        self.events = []
        self.steps += 1
        for b in self.bombers:
            b.update_cooldowns()
        for z in self.zombies:
            z.cooldown = max(0, z.cooldown - 1)
        for exp in self.explosions:
            exp.update()
        self.explosions = [e for e in self.explosions if e.active]

        for h in self.humans:
            if h.is_alive and not h.escaped:
                self._move(h, *ACTION_MAP.get(human_actions.get(h.id, 0), (0, 0)))
                if self.map.in_safe_zone(h.x, h.y, h.radius):
                    h.escaped = True
                    self.events.append(CombatEvent('human_escaped', target_id=h.id))

        self._update_targets()
        humans_by_id = {h.id: h for h in self.humans}
        for z in self.zombies:
            if z.target is None:
                continue
            target = humans_by_id[z.target]
            if z.distance_to(target) > z.attack_range:
                waypoint = self.pathfinder.get_next_waypoint((z.x, z.y), (target.x, target.y))
                wx, wy = waypoint or (target.x, target.y)
                self._move(z, wx-z.x, wy-z.y, slide=True)
        self._update_targets()

        # Each bomber resolves immediately, in stable ID order. A second bomber cannot
        # earn credit for damage or kills already caused by the first.
        for b in self.bombers:
            if not b.is_alive:
                continue
            action = bomber_actions.get(b.id, 0)
            if action in ACTION_MAP:
                self._move(b, *ACTION_MAP[action])
            elif action == 9:
                eligible = any(z.is_alive and z.is_threatening and b.distance_to(z) <= b.explosion_radius
                               for z in self.zombies)
                if b.can_explode() and eligible:
                    b.trigger_explosion()
                    exp = Explosion(b.x, b.y, b.explosion_radius,
                                    damage=b.explosion_damage, owner_id=b.id)
                    self.explosions.append(exp)
                    self.events.append(CombatEvent('explosion', actor_id=b.id))
                    self.events.extend(apply_explosion_damage(exp, self.zombies))
                else:
                    self.events.append(CombatEvent('invalid_explosion', actor_id=b.id))

        # Only surviving zombies may attack, so a same-step rescue is possible.
        for z in self.zombies:
            if not z.is_alive or z.target is None or z.cooldown > 0:
                continue
            h = humans_by_id[z.target]
            if h.is_alive and not h.escaped and z.distance_to(h) <= z.attack_range:
                damage = h.take_damage(z.attack_damage)
                z.cooldown = z.cooldown_max
                self.events.append(CombatEvent('human_damage', z.id, h.id, damage))
                if not h.is_alive:
                    self.events.append(CombatEvent('human_died', z.id, h.id))
        self._update_targets()
        self._check_outcome()
        return self.get_state()

    def _check_outcome(self):
        active = [h for h in self.humans if h.is_alive and not h.escaped]
        escaped = sum(h.escaped for h in self.humans)
        if not active:
            self.outcome = 'victory' if escaped else 'defeat'
        elif (self.config['end_on_zombie_clear'] and self.config['num_zombies'] > 0
              and not any(z.is_alive for z in self.zombies)):
            self.outcome = 'victory'
        elif self.steps >= self.config['max_steps']:
            self.truncated = True
