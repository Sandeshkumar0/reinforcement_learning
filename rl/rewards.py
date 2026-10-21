"""Event-based rewards with configurable, separately logged components."""
import math

HUMAN_WEIGHTS = dict(step=-0.01, progress=5.0, damage=-1.0, death=-100.0,
                     escape=100.0, danger=-0.05)
BOMBER_WEIGHTS = dict(step=-0.01, damage=0.1, kill=10.0, threat_damage=0.1,
                      threat_kill=10.0, human_damage=-0.2, human_death=-50.0,
                      human_escape=20.0, invalid=-1.0, progress=2.0)


def safe_distance(state, human):
    """Distance to nearest safe-zone boundary; zero once inside."""
    return min((math.hypot(max(sx-human.x, 0, human.x-(sx+sw)),
                           max(sy-human.y, 0, human.y-(sy+sh)))
                for sx, sy, sw, sh in state['map'].safe_zones), default=0.0)


def human_reward_components(prev_state, new_state, human_id=0, weights=None, gamma=0.99):
    w = {**HUMAN_WEIGHTS, **(weights or {})}
    prev = next((h for h in prev_state['humans'] if h.id == human_id), None)
    curr = next((h for h in new_state['humans'] if h.id == human_id), None)
    if not prev or not curr or not prev.is_alive or prev.escaped:
        return {key: 0.0 for key in w}
    events = [e for e in new_state['events'] if e.target_id == human_id]
    diag = math.hypot(new_state['map'].width, new_state['map'].height)
    # Bounded potential shaping with the same gamma as PPO prevents profitable loops.
    phi_prev = -safe_distance(prev_state, prev) / diag
    phi_new = -safe_distance(new_state, curr) / diag
    if new_state['outcome'] is not None:
        phi_new = 0.0
    danger = max((max(0.0, 1-curr.distance_to(z)/80)
                  for z in new_state['zombies'] if z.is_alive), default=0.0)
    return dict(step=w['step'], progress=w['progress']*(gamma*phi_new-phi_prev),
                damage=w['damage']*sum(e.amount for e in events if e.kind == 'human_damage'),
                death=w['death']*sum(e.kind == 'human_died' for e in events),
                escape=w['escape']*sum(e.kind == 'human_escaped' for e in events),
                danger=w['danger']*danger if curr.is_alive and not curr.escaped else 0.0)


def _threat_potential(state, bomber_id):
    b = next((b for b in state['bombers'] if b.id == bomber_id), None)
    threats = [z for z in state['zombies'] if z.is_alive and z.is_threatening]
    if not b or not threats or state['outcome'] is not None:
        return 0.0
    diag = math.hypot(state['map'].width, state['map'].height)
    return -min(b.distance_to(z) for z in threats) / diag


def bomber_reward_components(prev_state, new_state, bomber_id=0, weights=None, gamma=0.99):
    w = {**BOMBER_WEIGHTS, **(weights or {})}
    own = [e for e in new_state['events'] if e.actor_id == bomber_id
           and e.kind in ('zombie_damage', 'zombie_killed', 'explosion', 'invalid_explosion')]
    events = new_state['events']
    return dict(step=w['step'],
                damage=w['damage']*sum(e.amount for e in own if e.kind == 'zombie_damage'),
                kill=w['kill']*sum(e.kind == 'zombie_killed' for e in own),
                threat_damage=w['threat_damage']*sum(e.amount for e in own if e.kind == 'zombie_damage' and e.threatening),
                threat_kill=w['threat_kill']*sum(e.kind == 'zombie_killed' and e.threatening for e in own),
                human_damage=w['human_damage']*sum(e.amount for e in events if e.kind == 'human_damage'),
                human_death=w['human_death']*sum(e.kind == 'human_died' for e in events),
                human_escape=w['human_escape']*sum(e.kind == 'human_escaped' for e in events),
                invalid=w['invalid']*sum(e.kind == 'invalid_explosion' for e in own),
                progress=w['progress']*(gamma*_threat_potential(new_state, bomber_id)
                                        - _threat_potential(prev_state, bomber_id)))


def compute_human_reward(prev_state, new_state, **kwargs):
    return sum(human_reward_components(prev_state, new_state, **kwargs).values()) if prev_state and new_state else 0.0


def compute_bomber_reward(prev_state, new_state, **kwargs):
    return sum(bomber_reward_components(prev_state, new_state, **kwargs).values()) if prev_state and new_state else 0.0
