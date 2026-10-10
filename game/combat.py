import math

class Explosion:
    def __init__(self, x, y, radius, duration=10):
        self.x = x
        self.y = y
        self.radius = radius
        self.duration = duration # Frames it stays active visually
        self.active = True

    def update(self):
        self.duration -= 1
        if self.duration <= 0:
            self.active = False

class Arrow:
    def __init__(self, x, y, dx, dy, speed=5.0):
        self.x = x
        self.y = y
        # normalize dir
        length = math.hypot(dx, dy)
        if length > 0:
            self.dx = dx / length
            self.dy = dy / length
        else:
            self.dx = 1
            self.dy = 0
        self.speed = speed
        self.active = True

    def move(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed

def apply_explosion_damage(explosion, zombies):
    killed = 0
    for zombie in zombies:
        if zombie.is_alive:
            dist = math.hypot(zombie.x - explosion.x, zombie.y - explosion.y)
            if dist <= explosion.radius + zombie.radius:
                zombie.is_alive = False
                killed += 1
    return killed

def check_arrow_collisions(arrow, zombies, game_map):
    # Check map bounds and obstacles
    if game_map.is_collision(arrow.x, arrow.y, radius=1):
        arrow.active = False
        return False
        
    for zombie in zombies:
        if zombie.is_alive:
            dist = math.hypot(zombie.x - zombie.x, zombie.y - zombie.y)
            # simpler check
            dist = math.hypot(arrow.x - zombie.x, arrow.y - zombie.y)
            if dist <= zombie.radius:
                zombie.is_alive = False
                arrow.active = False
                return True
    return False
