import random

class Map:
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.obstacles = [] # List of (x, y, w, h)
        self.safe_zones = [] # List of (x, y, w, h)

    def generate_random(self, num_obstacles=10, num_safe_zones=1, rng=None):
        rng = rng or random
        self.obstacles = []
        for _ in range(num_obstacles):
            w = rng.randint(30, 100)
            h = rng.randint(30, 100)
            x = rng.randint(0, self.width - w)
            y = rng.randint(0, self.height - h)
            self.obstacles.append((x, y, w, h))

        self.safe_zones = []
        for _ in range(num_safe_zones):
            w = h = 80
            for attempt in range(10000):
                x = rng.randint(0, self.width - w)
                y = rng.randint(0, self.height - h)
                if not any(x < ox+ow+10 and x+w+10 > ox and y < oy+oh+10 and y+h+10 > oy
                           for ox, oy, ow, oh in self.obstacles + self.safe_zones):
                    self.safe_zones.append((x, y, w, h))
                    break
            else:
                raise ValueError('Cannot place safe zone; reduce obstacles or increase map size.')

    def is_collision(self, x, y, radius):
        # Check boundaries
        if x - radius < 0 or x + radius > self.width:
            return True
        if y - radius < 0 or y + radius > self.height:
            return True

        # Check obstacles
        for ox, oy, ow, oh in self.obstacles:
            # Simple AABB vs Circle collision check
            test_x = x
            test_y = y
            
            if x < ox: test_x = ox
            elif x > ox + ow: test_x = ox + ow
            
            if y < oy: test_y = oy
            elif y > oy + oh: test_y = oy + oh
            
            dist_x = x - test_x
            dist_y = y - test_y
            distance = (dist_x**2 + dist_y**2) ** 0.5
            
            if distance <= radius:
                return True
        return False

    def in_safe_zone(self, x, y, radius):
        for sx, sy, sw, sh in self.safe_zones:
            # If center is in the box
            if sx <= x <= sx + sw and sy <= y <= sy + sh:
                return True
        return False
