import pygame

class Renderer:
    def __init__(self, config):
        pygame.init()
        self.width = config['width']
        self.height = config['height']
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("RL Zombie Survival")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)
        self.fps = 60

    def draw_state(self, state):
        self.screen.fill((220, 220, 220)) # Light gray background
        
        # Draw map obstacles
        for ox, oy, ow, oh in state['map'].obstacles:
            pygame.draw.rect(self.screen, (100, 100, 100), (ox, oy, ow, oh))
            
        # Draw safe zones
        for sx, sy, sw, sh in state['map'].safe_zones:
            pygame.draw.rect(self.screen, (144, 238, 144), (sx, sy, sw, sh))
            # Safe zone label
            lbl = self.font.render("SAFE", True, (0, 100, 0))
            self.screen.blit(lbl, (sx + sw//2 - 20, sy + sh//2 - 10))

        # Threat lines identify the human a zombie is trying to attack.
        humans_by_id = {h.id: h for h in state['humans']}
        for z in state['zombies']:
            target = humans_by_id.get(z.target)
            if z.is_alive and z.is_threatening and target:
                pygame.draw.line(self.screen, (210, 120, 120),
                                 (int(z.x), int(z.y)), (int(target.x), int(target.y)), 1)

        # Draw entities
        for h in state['humans']:
            if h.is_alive and not h.escaped:
                pygame.draw.circle(self.screen, h.color, (int(h.x), int(h.y)), h.radius)
                self.draw_health(h)
                
        for z in state['zombies']:
            if z.is_alive:
                pygame.draw.circle(self.screen, z.color, (int(z.x), int(z.y)), z.radius)
                self.draw_health(z)
                
        for b in state['bombers']:
            if b.is_alive:
                pygame.draw.circle(self.screen, b.color, (int(b.x), int(b.y)), b.radius)
                # Draw activation radius faintly
                pygame.draw.circle(self.screen, (255, 200, 0), (int(b.x), int(b.y)), b.activation_radius, 1)

        # Draw explosions
        for exp in state['explosions']:
            pygame.draw.circle(self.screen, (255, 255, 0), (int(exp.x), int(exp.y)), exp.radius)

        # Draw UI text
        ui_text = [
            f"Steps: {state['steps']}",
            f"Humans Alive/Escaped: {sum(1 for h in state['humans'] if h.is_alive)} / {sum(1 for h in state['humans'] if h.escaped)}",
            f"Zombies Alive: {sum(1 for z in state['zombies'] if z.is_alive)}",
        ]
        
        if state.get('truncated'):
            ui_text.append('Outcome: TIMEOUT')
        if state['bombers']:
            b = state['bombers'][0]
            ui_text.append(f"Bomber: {'READY' if b.cooldown == 0 else str(b.cooldown) + ' steps cooldown'}")
        if state['outcome']:
            ui_text.append(f"Outcome: {state['outcome'].upper()}")
            
        for i, text in enumerate(ui_text):
            surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(surface, (10, 10 + i * 25))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def draw_health(self, entity):
        x, y = int(entity.x)-10, int(entity.y)-entity.radius-7
        pygame.draw.rect(self.screen, (80, 80, 80), (x, y, 20, 3))
        pygame.draw.rect(self.screen, (0, 170, 40),
                         (x, y, round(20*entity.health/entity.max_health), 3))

    def process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True
