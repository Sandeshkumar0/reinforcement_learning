"""Frozen-policy helpers and a deterministic, obstacle-aware human fallback."""
import math
from pathlib import Path
from game.engine import ACTION_MAP
from game.observations import OBS_DIM


def load_policy(path, actions):
    from stable_baselines3 import PPO
    if not Path(path).is_file() and not Path(str(path)+'.zip').is_file():
        raise FileNotFoundError(f'Checkpoint not found: {path}')
    model = PPO.load(path, device='cpu')
    if model.observation_space.shape != (OBS_DIM,) or model.action_space.n != actions:
        raise ValueError('Checkpoint uses an incompatible observation/action schema. '
                         'Retrain with schema v2; old checkpoints cannot be resumed.')
    return model


def fallback_human_action(engine, human):
    """Choose a collision-free step toward an A* waypoint, avoiding nearby zombies."""
    if not human.is_alive or human.escaped:
        return 0
    zones = engine.map.safe_zones
    goal = min(((sx+sw/2, sy+sh/2) for sx, sy, sw, sh in zones),
               key=lambda p: math.hypot(p[0]-human.x, p[1]-human.y), default=(human.x, human.y))
    waypoint = engine.pathfinder.get_next_waypoint((human.x, human.y), goal) or goal
    threats = [z for z in engine.zombies if z.is_alive]
    best = (-float('inf'), 0)
    for action, (dx, dy) in ACTION_MAP.items():
        norm = math.hypot(dx, dy) or 1
        x, y = human.x+dx/norm*human.speed, human.y+dy/norm*human.speed
        if engine.map.is_collision(x, y, human.radius):
            continue
        score = -math.hypot(x-waypoint[0], y-waypoint[1])
        score -= sum(max(0, 80-math.hypot(x-z.x, y-z.y))*3 for z in threats)
        if score > best[0]:
            best = (score, action)
    return best[1]
