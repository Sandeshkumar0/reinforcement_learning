"""Behavioral regressions for combat, credit assignment, observations and PPO."""
import copy
import warnings
import numpy as np
import pytest
from stable_baselines3.common.env_checker import check_env
from game.engine import GameEngine
from game.environment import ZombieSurvivalEnv, BomberCombatEnv
from game.combat import Explosion, apply_explosion_damage
from game.observations import observation, OBS_DIM
from rl.rewards import human_reward_components, bomber_reward_components


def scenario(**overrides):
    engine = GameEngine(dict(width=300, height=300, num_humans=1, num_zombies=1,
        num_bombers=1, num_obstacles=0, num_safe_zones=0, max_steps=100,
        zombie_speed=0, human_speed=2.5, end_on_zombie_clear=False, **overrides))
    h, z, b = engine.humans[0], engine.zombies[0], engine.bombers[0]
    h.x, h.y = 100, 100
    z.x, z.y = 110, 100
    b.x, b.y = 120, 100
    engine._update_targets()
    return engine


def test_detached_state_and_human_damage_death_rewards():
    engine = scenario(human_health=25)
    previous = engine.get_state()
    current = engine.step()
    assert previous['humans'][0].health == 25
    assert previous['humans'][0].is_alive
    assert not current['humans'][0].is_alive
    parts = human_reward_components(previous, current)
    assert parts['damage'] == -25
    assert parts['death'] == -100
    current['humans'][0].x = 999
    assert engine.humans[0].x == 100
    # Returned combat events are immutable.
    with pytest.raises(AttributeError):
        current['events'][0].amount = 0
    assert engine.step()['events'] == ()


def test_human_escape_reward_and_outcome_with_live_zombie():
    engine = scenario()
    engine.map.safe_zones = [(95, 95, 20, 20)]
    previous = engine.get_state()
    current = engine.step()
    assert current['humans'][0].escaped and current['outcome'] == 'victory'
    assert current['zombies'][0].is_alive
    assert human_reward_components(previous, current)['escape'] == 100
    assert not any(e.kind == 'human_damage' for e in current['events'])


def test_progress_reward_and_configurable_weights():
    engine = scenario()
    engine.zombies[0].x = 280
    engine.map.safe_zones = [(200, 100, 50, 50)]
    previous = engine.get_state()
    current = engine.step({0: 3})
    p = human_reward_components(previous, current, weights={'step': -2, 'progress': 3}, gamma=1)
    assert p['progress'] > 0 and p['step'] == -2
    stayed = copy.deepcopy(previous)
    assert human_reward_components(previous, stayed, gamma=1)['progress'] == 0


def test_partial_damage_rewards_then_capped_kill_and_no_repeated_damage():
    engine = scenario(zombie_health=150, explosion_damage=100, explosion_cooldown=2)
    previous = engine.get_state()
    current = engine.step(bomber_actions={0: 9})
    p = bomber_reward_components(previous, current)
    assert p['damage'] == 10 and p['threat_damage'] == 10 and p['kill'] == 0
    assert engine.zombies[0].health == 50
    current = engine.step(bomber_actions={0: 9})  # Still cooling down.
    assert engine.zombies[0].health == 50
    assert bomber_reward_components(previous, current)['invalid'] == -1
    previous = current
    current = engine.step(bomber_actions={0: 9})
    p = bomber_reward_components(previous, current)
    assert p['damage'] == 5 and p['kill'] == 10 and p['threat_kill'] == 10
    assert sum(e.amount for e in current['events'] if e.kind == 'zombie_damage') == 50
    current = engine.step()
    assert not any(e.kind.startswith('zombie_') for e in current['events'])


def test_explosion_visual_lifetime_never_damages_new_arrivals():
    engine = scenario()
    z = engine.zombies[0]
    z.x = 250
    exp = Explosion(100, 100, 70)
    assert apply_explosion_damage(exp, [z]) == []
    z.x = 110
    assert apply_explosion_damage(exp, [z]) == []
    assert z.health == z.max_health


def test_same_step_lethal_attack_prevented_by_rescue():
    engine = scenario(human_health=25)
    current = engine.step(bomber_actions={0: 9})
    assert current['humans'][0].health == 25
    assert not current['zombies'][0].is_alive
    assert not any(e.kind == 'human_damage' for e in current['events'])
    assert current['bombers'][0].is_alive


@pytest.mark.parametrize('case', ['outside_blast', 'no_threat', 'dead_zombie', 'escaped_human', 'dead_human'])
def test_explosion_eligibility(case):
    engine = scenario()
    if case == 'outside_blast':
        engine.bombers[0].x = 250
    elif case == 'no_threat':
        engine.humans[0].x = 260
    elif case == 'dead_zombie':
        engine.zombies[0].take_damage(100)
    elif case == 'escaped_human':
        engine.humans[0].escaped = True
    else:
        engine.humans[0].take_damage(100)
    current = engine.step(bomber_actions={0: 9})
    assert any(e.kind == 'invalid_explosion' for e in current['events'])
    assert not engine.explosions and engine.bombers[0].cooldown == 0


def test_boundary_of_blast_and_threat_is_inclusive():
    engine = scenario()
    engine.humans[0].x = 210  # Exactly threat_distance from zombie.
    engine.bombers[0].x = 180  # Exactly explosion_radius from zombie.
    current = engine.step(bomber_actions={0: 9})
    assert any(e.kind == 'explosion' for e in current['events'])


def test_cooldowns_advance_even_without_actions():
    engine = scenario(explosion_damage=10, explosion_cooldown=3, zombie_attack_cooldown=3)
    engine.step(bomber_actions={0: 9})
    assert engine.humans[0].health == 75
    assert engine.bombers[0].cooldown == 3
    for _ in range(2):
        engine.step()
        assert engine.humans[0].health == 75
    engine.step()
    assert engine.humans[0].health == 50 and engine.bombers[0].cooldown == 0


def test_kill_credit_goes_only_to_responsible_bomber():
    engine = GameEngine(dict(num_humans=1, num_zombies=1, num_bombers=2,
                             num_obstacles=0, num_safe_zones=0, zombie_speed=0))
    for entity in engine.humans + engine.zombies + engine.bombers:
        entity.x, entity.y = 100, 100
    engine._update_targets()
    previous = engine.get_state()
    current = engine.step(bomber_actions={0: 9, 1: 9})
    assert bomber_reward_components(previous, current, 0)['kill'] == 10
    assert bomber_reward_components(previous, current, 1)['kill'] == 0
    assert bomber_reward_components(previous, current, 1)['damage'] == 0
    assert bomber_reward_components(previous, current, 1)['invalid'] == -1


def test_zombies_pursue_nearest_active_human_and_ignore_escaped():
    engine = GameEngine(dict(num_humans=2, num_zombies=1, num_obstacles=0,
                             num_safe_zones=0, zombie_speed=1.5))
    z = engine.zombies[0]
    z.x, z.y = 100, 100
    engine.humans[0].x, engine.humans[0].y = 110, 100
    engine.humans[0].escaped = True
    engine.humans[1].x, engine.humans[1].y = 180, 100
    before = z.distance_to(engine.humans[1])
    engine.step()
    assert z.target == 1 and z.is_threatening
    assert z.distance_to(engine.humans[1]) < before
    assert engine.humans[0].health == 100


def test_all_zombies_cleared_and_human_death_outcomes():
    engine = scenario()
    engine.config['end_on_zombie_clear'] = True
    assert engine.step(bomber_actions={0: 9})['outcome'] == 'victory'
    engine = scenario(human_health=25)
    assert engine.step()['outcome'] == 'defeat'


def test_navigation_zero_zombies_continues_and_timeout_truncates():
    env = ZombieSurvivalEnv(config={'num_zombies': 0, 'num_bombers': 0, 'num_obstacles': 0, 'max_steps': 2})
    env.reset(seed=10)
    _, _, terminated, truncated, _ = env.step(0)
    assert not terminated and not truncated
    _, _, terminated, truncated, info = env.step(0)
    assert not terminated and truncated
    assert info['episode_stats']['steps'] == 2
    with pytest.raises(RuntimeError):
        env.step(0)


@pytest.mark.parametrize('cls', [ZombieSurvivalEnv, BomberCombatEnv])
def test_seed_reproducibility_and_observation_masks(cls):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        env = cls(config={'num_obstacles': 2, 'max_steps': 8})
    first, _ = env.reset(seed=123)
    trajectory = [env.step(0) for _ in range(3)]
    second, _ = env.reset(seed=123)
    assert np.array_equal(first, second)
    for old in trajectory:
        new = env.step(0)
        assert np.array_equal(old[0], new[0])
        assert old[1:4] == new[1:4]
    state = env.game_engine.get_state()
    ids_before = [z.id for z in state['zombies']]
    obs = observation(state, state['humans'][0], 8)
    assert ids_before == [z.id for z in state['zombies']]
    assert obs.shape == (OBS_DIM,) and env.observation_space.contains(obs)
    assert obs[9] == 1  # Present human slot 0.
    assert np.array_equal(obs[15:21], np.zeros(6))  # Absent human slot 1.
    env.close()


def test_observation_does_not_reorder_zombies_and_exposes_threat():
    engine = scenario()
    state = engine.get_state()
    obs = observation(state, state['bombers'][0], 100)
    zombie_slot = 9+8*6
    assert obs[zombie_slot+6] == 1  # Valid human target, even though target ID is 0.
    assert obs[zombie_slot+7] == 1
    engine.humans[0].x = 280
    engine._update_targets()
    obs2 = observation(engine.get_state(), engine.bombers[0], 100)
    assert obs2[zombie_slot+7] == 0
    engine.map.obstacles = [(130, 90, 30, 30)]
    obs3 = observation(engine.get_state(), engine.bombers[0], 100)
    assert obs3[-9] < 0.1  # East ray sees obstacle just beside bomber.


def test_diagonal_collision_check_uses_actual_normalized_motion():
    engine = scenario()
    human = engine.humans[0]
    human.x = engine.map.width-human.radius-2
    engine._move(human, 1, 1)
    assert human.x > engine.map.width-human.radius-2
    assert not engine.map.is_collision(human.x, human.y, human.radius)


def test_environment():
    for cls in (ZombieSurvivalEnv, BomberCombatEnv):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            env = cls(config={'num_obstacles': 0})
        check_env(env, warn=True)
        env.close()
