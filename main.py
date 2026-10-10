import argparse
import random
import pygame
from game.engine import GameEngine
from ui.renderer import Renderer

def run_random_play():
    config = {
        'width': 800,
        'height': 600,
        'num_humans': 1,
        'num_zombies': 3,
        'num_bombers': 1,
        'num_obstacles': 15,
        'num_safe_zones': 1,
        'max_steps': 1000
    }
    
    engine = GameEngine(config)
    renderer = Renderer(config)
    
    running = True
    while running:
        running = renderer.process_events()
        
        # Generate random actions for testing
        human_actions = {0: random.randint(0, 8)}
        
        bomber_action = random.randint(0, 8)
        # 5% chance to explode if zombies nearby
        state = engine.get_state()
        b = state['bombers'][0]
        for z in state['zombies']:
            if z.is_alive and b.distance_to(z) <= b.activation_radius:
                if random.random() < 0.05:
                    bomber_action = 9
                break
                
        bomber_actions = {0: bomber_action}
        
        new_state = engine.step(human_actions, bomber_actions)
        renderer.draw_state(new_state)
        
        if new_state['outcome'] is not None:
            # wait 2 seconds then reset
            pygame.time.wait(2000)
            engine.reset()

def main():
    parser = argparse.ArgumentParser(description="RL Zombie Survival Game")
    parser.add_argument('--play', action='store_true', help='Run simulation with random actions')
    parser.add_argument('--train-human', action='store_true', help='Train the human agent using PPO')
    parser.add_argument('--evaluate-human', action='store_true', help='Evaluate trained human agent')
    parser.add_argument('--train-bomber', action='store_true', help='Train the bomber agent using PPO')
    parser.add_argument('--evaluate-bomber', action='store_true', help='Evaluate trained bomber agent')
    args = parser.parse_args()
    
    if args.play:
        run_random_play()
    elif args.train_human:
        from rl.train_human import train_human
        train_human()
    elif args.evaluate_human:
        from rl.evaluate import evaluate
        evaluate("human", "models/human/human_final_phase_3")
    elif args.train_bomber:
        from rl.train_bomber import train_bomber
        train_bomber()
    elif args.evaluate_bomber:
        from rl.evaluate import evaluate
        evaluate("bomber", "models/bomber/bomber_final")
    else:
        print("Please specify a mode. Example: --play, --train-human, --train-bomber, --evaluate-human, --evaluate-bomber")

if __name__ == "__main__":
    main()
