"""One-shot combat effects and immutable, attributable step events."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class CombatEvent:
    kind: str
    actor_id: int = -1
    target_id: int = -1
    amount: float = 0.0
    threatening: bool = False


class Explosion:
    def __init__(self, x, y, radius, duration=10, damage=100, owner_id=0):
        self.x, self.y, self.radius = x, y, radius
        self.duration, self.damage, self.owner_id = duration, damage, owner_id
        self.active = True
        self.damage_applied = False

    def update(self):
        self.duration -= 1
        self.active = self.duration > 0


def apply_explosion_damage(explosion, zombies):
    """Return damage/kill events once; visual duration never causes more damage."""
    if explosion.damage_applied:
        return []
    explosion.damage_applied = True
    events = []
    for zombie in zombies:
        if zombie.is_alive and math.hypot(zombie.x - explosion.x, zombie.y - explosion.y) <= explosion.radius:
            damage = zombie.take_damage(explosion.damage)
            if damage:
                events.append(CombatEvent('zombie_damage', explosion.owner_id, zombie.id,
                                          damage, zombie.is_threatening))
                if not zombie.is_alive:
                    events.append(CombatEvent('zombie_killed', explosion.owner_id, zombie.id,
                                              threatening=zombie.is_threatening))
    return events
