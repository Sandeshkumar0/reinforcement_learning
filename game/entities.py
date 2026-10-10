import math

class Entity:
    def __init__(self, x, y, radius, speed, color):
        self.x = x
        self.y = y
        self.radius = radius
        self.speed = speed
        self.color = color
        self.is_alive = True

    def distance_to(self, other_entity):
        return math.hypot(self.x - other_entity.x, self.y - other_entity.y)

    def move(self, dx, dy):
        # Normalize direction
        length = math.hypot(dx, dy)
        if length > 0:
            dx /= length
            dy /= length
        self.x += dx * self.speed
        self.y += dy * self.speed


class Human(Entity):
    def __init__(self, x, y, radius=5, speed=2.5, color=(0, 0, 255)): # Blue
        super().__init__(x, y, radius, speed, color)
        self.escaped = False

class Zombie(Entity):
    def __init__(self, x, y, radius=5, speed=1.5, color=(255, 0, 0)): # Red
        super().__init__(x, y, radius, speed, color)
        self.target = None

class Bomber(Entity):
    def __init__(self, x, y, radius=6, speed=2.0, color=(255, 165, 0)): # Orange
        super().__init__(x, y, radius, speed, color)
        self.activation_radius = 100
        self.explosion_radius = 70
        self.cooldown_max = 60 # frames/steps
        self.cooldown = 0
        self.arrows_cooldown_max = 30
        self.arrows_cooldown = 0

    def can_explode(self):
        return self.cooldown <= 0

    def trigger_explosion(self):
        if self.can_explode():
            self.cooldown = self.cooldown_max
            return True
        return False

    def update_cooldowns(self):
        if self.cooldown > 0:
            self.cooldown -= 1
        if self.arrows_cooldown > 0:
            self.arrows_cooldown -= 1
