def compute_human_reward(prev_state, new_state):
    """
    Computes reward for the human agent based on state transition.
    """
    if not prev_state or not new_state:
        return 0.0
        
    human_prev = prev_state['humans'][0]
    human_new = new_state['humans'][0]
    
    # Base survival reward
    reward = 0.5
    
    # Check if killed
    if human_prev.is_alive and not human_new.is_alive:
        reward -= 100.0
        return reward
        
    # Check if escaped
    if not human_prev.escaped and human_new.escaped:
        reward += 100.0
        # Bonus for fewer steps
        reward += max(0, 1000 - new_state['steps']) * 0.1
        return reward

    # Reward moving closer to safe zone
    sz_x, sz_y = 0, 0
    if new_state['map'].safe_zones:
        sx, sy, sw, sh = new_state['map'].safe_zones[0]
        sz_x, sz_y = sx + sw/2, sy + sh/2
        
        dist_prev = ((human_prev.x - sz_x)**2 + (human_prev.y - sz_y)**2)**0.5
        dist_new = ((human_new.x - sz_x)**2 + (human_new.y - sz_y)**2)**0.5
        
        # Small reward for moving closer, penalty for moving away
        reward += (dist_prev - dist_new) * 0.1

    # Penalize if too close to zombies to encourage dodging
    for z in new_state['zombies']:
        if z.is_alive:
            dist = ((human_new.x - z.x)**2 + (human_new.y - z.y)**2)**0.5
            if dist < 80:
                reward -= (80 - dist) * 0.1
                
    return reward

def compute_bomber_reward(prev_state, new_state):
    """
    Computes reward for the bomber agent.
    Encourages killing zombies efficiently and penalizes wasted explosions.
    """
    if not prev_state or not new_state:
        return 0.0
        
    reward = 0.0
    
    # Count zombies killed in this step
    prev_zombies_alive = sum(1 for z in prev_state['zombies'] if z.is_alive)
    new_zombies_alive = sum(1 for z in new_state['zombies'] if z.is_alive)
    killed = prev_zombies_alive - new_zombies_alive
    
    if killed > 0:
        # Exponential reward for multi-kills (1 kill = 10, 2 kills = 30, 3 kills = 70)
        reward += (2 ** killed - 1) * 10.0
        
    # Penalize wasted explosion slightly (don't penalize too much to allow exploration)
    prev_bomber = prev_state['bombers'][0]
    new_bomber = new_state['bombers'][0]
    
    # If cooldown went from 0 to max, it means we exploded
    if prev_bomber.cooldown == 0 and new_bomber.cooldown > 0:
        if killed == 0:
            reward -= 0.1 # Reduced wasted attack penalty

    # Shaping reward: get closer to zombies
    prev_zombies = [z for z in prev_state['zombies'] if z.is_alive]
    new_zombies = [z for z in new_state['zombies'] if z.is_alive]
    if prev_zombies and new_zombies:
        dist_prev = min([prev_bomber.distance_to(z) for z in prev_zombies])
        dist_new = min([new_bomber.distance_to(z) for z in new_zombies])
        # Reward getting closer to the nearest zombie
        reward += (dist_prev - dist_new) * 0.5

    # Small penalty for every step to encourage action
    reward -= 0.01

    return reward
