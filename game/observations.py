"""Observation schema v2: fixed ID slots and masks; all values in [-1, 1]."""
import math
import numpy as np

MAX_HUMANS, MAX_ZOMBIES, MAX_BOMBERS = 8, 12, 4
# Self x/y, health, cooldown, blast radius; goal dx/dy/distance/mask;
# humans [present, alive, escaped, x, y, health];
# zombies [present, alive, x, y, health, target-id, target-valid, threat, cooldown];
# bombers [present, alive, x, y, cooldown]; 8 collision rays; remaining time.
OBS_DIM = 5 + 4 + MAX_HUMANS*6 + MAX_ZOMBIES*9 + MAX_BOMBERS*5 + 8 + 1


def validate_capacity(config):
    for key, cap in [('num_humans', MAX_HUMANS), ('num_zombies', MAX_ZOMBIES), ('num_bombers', MAX_BOMBERS)]:
        if config[key] > cap:
            raise ValueError(f'{key} exceeds observation capacity {cap}')


def collision_rays(game_map, entity, max_distance=100):
    rays = []
    for angle in np.arange(8) * math.pi / 4:
        free = 0.0
        for distance in range(5, max_distance+1, 5):
            if game_map.is_collision(entity.x+math.cos(angle)*distance,
                                     entity.y+math.sin(angle)*distance, entity.radius):
                break
            free = distance / max_distance
        rays.append(free)
    return rays


def observation(state, entity, max_steps):
    if entity is None:
        return np.zeros(OBS_DIM, dtype=np.float32)
    width, height = state['map'].width, state['map'].height
    diag = math.hypot(width, height)
    cooldown = lambda e: getattr(e, 'cooldown', 0) / max(1, getattr(e, 'cooldown_max', 1))
    obs = [entity.x/width, entity.y/height, entity.health/entity.max_health,
           cooldown(entity), getattr(entity, 'explosion_radius', 0)/diag]
    zones = state['map'].safe_zones
    if zones:
        sx, sy, sw, sh = min(zones, key=lambda s: math.hypot(s[0]+s[2]/2-entity.x, s[1]+s[3]/2-entity.y))
        dx, dy = sx+sw/2-entity.x, sy+sh/2-entity.y
        obs.extend([dx/width, dy/height, math.hypot(dx, dy)/diag, 1])
    else:
        obs.extend([0]*4)
    humans = {h.id: h for h in state['humans']}
    for slot in range(MAX_HUMANS):
        h = humans.get(slot)
        obs.extend([1, h.is_alive, h.escaped, h.x/width, h.y/height, h.health/h.max_health] if h else [0]*6)
    zombies = {z.id: z for z in state['zombies']}
    for slot in range(MAX_ZOMBIES):
        z = zombies.get(slot)
        obs.extend([1, z.is_alive, z.x/width, z.y/height, z.health/z.max_health,
                    z.target/(MAX_HUMANS-1) if z.target is not None else 0,
                    z.target is not None, z.is_threatening, cooldown(z)] if z else [0]*9)
    bombers = {b.id: b for b in state['bombers']}
    for slot in range(MAX_BOMBERS):
        b = bombers.get(slot)
        obs.extend([1, b.is_alive, b.x/width, b.y/height, cooldown(b)] if b else [0]*5)
    obs.extend(collision_rays(state['map'], entity))
    obs.append(max(0, 1-state['steps']/max_steps))
    return np.asarray(obs, dtype=np.float32)
