"""Simulation entities. IDs remain stable for the lifetime of an episode."""
import math


class Entity:
    def __init__(self, x, y, radius, speed, color, entity_id=0, health=100):
        self.x, self.y = float(x), float(y)
        self.radius, self.speed, self.color = radius, speed, color
        self.id = entity_id
        self.max_health = float(health)
        self.health = float(health)
        self.is_alive = True

    def distance_to(self, other):
        return math.hypot(self.x - other.x, self.y - other.y)

    def move(self, dx, dy):
        length = math.hypot(dx, dy)
        if length:
            self.x += dx / length * self.speed
            self.y += dy / length * self.speed

    def take_damage(self, amount):
        actual = min(self.health, max(0.0, amount)) if self.is_alive else 0.0
        self.health -= actual
        if self.health <= 0:
            self.is_alive = False
        return actual


class Human(Entity):
    def __init__(self, x, y, radius=5, speed=2.5, color=(0, 0, 255), **kwargs):
        super().__init__(x, y, radius, speed, color, **kwargs)
        self.escaped = False


class Zombie(Entity):
    def __init__(self, x, y, radius=5, speed=1.5, color=(255, 0, 0),
                 attack_damage=25, attack_range=12, attack_cooldown=30, **kwargs):
        super().__init__(x, y, radius, speed, color, **kwargs)
        self.target = None  # Human ID, never an object reference
        self.is_threatening = False
        self.attack_damage = attack_damage
        self.attack_range = attack_range  # Center-to-center distance
        self.cooldown_max = attack_cooldown
        self.cooldown = 0


class Bomber(Entity):
    def __init__(self, x, y, radius=6, speed=2.0, color=(255, 165, 0),
                 explosion_radius=70, explosion_damage=100, explosion_cooldown=60, **kwargs):
        super().__init__(x, y, radius, speed, color, **kwargs)
        self.explosion_radius = explosion_radius
        self.activation_radius = explosion_radius
        self.explosion_damage = explosion_damage
        self.cooldown_max = explosion_cooldown
        self.cooldown = 0

    def can_explode(self):
        return self.is_alive and self.cooldown == 0

    def trigger_explosion(self):
        if self.can_explode():
            self.cooldown = self.cooldown_max
            return True
        return False

    def update_cooldowns(self):
        self.cooldown = max(0, self.cooldown - 1)
