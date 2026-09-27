import numpy as np
import pytest

from traffic_rl import EW_MAX, NS_MAX, TrafficIntersectionEnv
from traffic_rl.agents import (Hyperparams, evaluate, greedy_policy, policy_from_table,
                               q_learning, value_iteration)
from traffic_rl.env import transition_model


def test_spaces_and_reset():
    env = TrafficIntersectionEnv()
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    obs, _ = env.reset(options={"state": (2, 4)})
    assert tuple(obs) == (2, 4)


def test_episode_truncates_after_50_steps():
    env = TrafficIntersectionEnv()
    env.reset(seed=1)
    for t in range(50):
        obs, r, term, trunc, info = env.step(env.action_space.sample())
        assert 0 <= obs[0] <= NS_MAX and 0 <= obs[1] <= EW_MAX
        assert r == -(obs[0] + obs[1])
        assert not term
    assert trunc


@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_transition_model_is_a_distribution(penalty):
    P = transition_model(penalty)
    assert len(P) == (NS_MAX + 1) * (EW_MAX + 1) * 2
    for outs in P.values():
        assert abs(sum(p for p, _, _ in outs) - 1) < 1e-12


def test_simulator_matches_transition_model():
    """Empirical next-state frequencies agree with the exact model."""
    P = transition_model()
    env = TrafficIntersectionEnv()
    env.reset(seed=3)
    s, a, n = (2, 3), 1, 20000
    counts = {}
    for _ in range(n):
        env.reset(options={"state": s})
        s2, *_ = env.step(a)
        counts[tuple(s2)] = counts.get(tuple(s2), 0) + 1
    for p, s2, _ in P[(*s, a)]:
        assert abs(counts.get(s2, 0) / n - p) < 0.015


def test_overflow_penalty_counts_dropped_cars():
    env = TrafficIntersectionEnv(overflow_penalty=2.0)
    env.reset(seed=0, options={"state": (NS_MAX, EW_MAX)})
    for _ in range(30):
        _, r, _, _, info = env.step(0)
        assert r == -info["waiting"] - 2.0 * info["dropped"]


def test_value_iteration_beats_simple_rules():
    pi = greedy_policy(value_iteration())
    opt, _ = evaluate(policy_from_table(pi), episodes=300)
    lqf, _ = evaluate(lambda s, t: int(s[0] > s[1]), episodes=300)
    assert opt < lqf


def test_q_learning_learns_something():
    hp = Hyperparams(episodes=400)
    Q, rewards = q_learning(hp, seed=0)
    assert rewards[-50:].mean() > rewards[:50].mean()
    assert np.isfinite(Q).all()
