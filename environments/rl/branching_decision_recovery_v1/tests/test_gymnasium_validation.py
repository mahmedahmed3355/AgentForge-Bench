from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from branching_decision_recovery_v1.actions import ActionKind
from branching_decision_recovery_v1.gymnasium_env import (
    BranchingDecisionRecoveryGymEnv,
)


def make_env() -> BranchingDecisionRecoveryGymEnv:
    return BranchingDecisionRecoveryGymEnv()


def test_gymnasium_check_env():
    env = make_env()
    try:
        check_env(env, skip_render_check=True)
    finally:
        env.close()


def test_reset_returns_valid_observation_and_info():
    env = make_env()

    observation, info = env.reset(seed=3001)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)

    env.close()


def test_reset_is_deterministic_for_same_seed():
    env1 = make_env()
    env2 = make_env()

    obs1, info1 = env1.reset(seed=3001)
    obs2, info2 = env2.reset(seed=3001)

    assert obs1.keys() == obs2.keys()

    for key in obs1:
        value1 = obs1[key]
        value2 = obs2[key]

        if isinstance(value1, np.ndarray):
            np.testing.assert_array_equal(value1, value2)
        else:
            assert value1 == value2

    assert info1 == info2

    env1.close()
    env2.close()


def test_different_seeds_are_accepted():
    env = make_env()

    obs1, _ = env.reset(seed=3001)
    obs2, _ = env.reset(seed=3002)

    assert env.observation_space.contains(obs1)
    assert env.observation_space.contains(obs2)

    env.close()


def test_action_space_matches_action_kind_count():
    env = make_env()

    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert env.action_space.n == len(ActionKind)

    env.close()


def test_every_action_index_is_valid_for_native_mapping():
    env = make_env()
    env.reset(seed=3001)

    for action_index in range(env.action_space.n):
        action_kind = list(ActionKind)[action_index]
        assert action_kind.value in {
            action.value for action in ActionKind
        }

    env.close()


def test_step_returns_standard_gymnasium_tuple():
    env = make_env()
    env.reset(seed=3001)

    observation, reward, terminated, truncated, info = env.step(0)

    assert env.observation_space.contains(observation)
    assert isinstance(reward, (int, float))
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)

    env.close()


def test_step_observation_remains_inside_space():
    env = make_env()
    env.reset(seed=3001)

    terminated = False
    truncated = False

    for _ in range(5):
        if terminated or truncated:
            break

        observation, reward, terminated, truncated, info = env.step(0)

        assert env.observation_space.contains(observation)

    env.close()


def test_reset_after_episode_restores_valid_interface():
    env = make_env()

    env.reset(seed=3001)

    terminated = False
    truncated = False

    for _ in range(10):
        if terminated or truncated:
            break
        _, _, terminated, truncated, _ = env.step(0)

    observation, info = env.reset(seed=3001)

    assert env.observation_space.contains(observation)
    assert isinstance(info, dict)
    assert terminated is False or truncated is False
    assert env.terminated is False
    assert env.truncated is False

    env.close()
