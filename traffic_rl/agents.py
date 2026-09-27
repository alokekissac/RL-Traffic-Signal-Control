"""Tabular agents (Q-learning, first-visit Monte Carlo), exact value iteration and baselines."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .env import EW_GREEN, EW_MAX, NS_GREEN, NS_MAX, TrafficIntersectionEnv, transition_model


@dataclass
class Hyperparams:
    alpha: float = 0.1            # learning rate (Q-learning)
    gamma: float = 0.9            # discount factor
    epsilon: float = 1.0          # initial exploration rate
    epsilon_decay: float = 0.995  # multiplicative decay per episode
    epsilon_min: float = 0.01
    episodes: int = 5000
    overflow_penalty: float = 0.0  # 0 = original coursework reward


def _new_q():
    return np.zeros((NS_MAX + 1, EW_MAX + 1, 2))


def _eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(0, 2))
    q = Q[s[0], s[1]]
    # break ties randomly so an all-zero row isn't always "EW green"
    return int(rng.choice(np.flatnonzero(q == q.max())))


def q_learning(hp: Hyperparams, seed: int = 0):
    """Off-policy TD control. Returns (Q, per-episode total rewards)."""
    rng = np.random.default_rng(seed)
    env = TrafficIntersectionEnv(overflow_penalty=hp.overflow_penalty)
    Q, rewards, eps = _new_q(), [], hp.epsilon
    for ep in range(hp.episodes):
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        total, done = 0.0, False
        while not done:
            a = _eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            target = r + (0.0 if term else hp.gamma * Q[s2[0], s2[1]].max())
            Q[s[0], s[1], a] += hp.alpha * (target - Q[s[0], s[1], a])
            s, total, done = s2, total + r, term or trunc
        rewards.append(total)
        eps = max(hp.epsilon_min, eps * hp.epsilon_decay)
    return Q, np.array(rewards)


def monte_carlo(hp: Hyperparams, seed: int = 0):
    """On-policy first-visit Monte Carlo control with sample-average returns."""
    rng = np.random.default_rng(seed)
    env = TrafficIntersectionEnv(overflow_penalty=hp.overflow_penalty)
    Q, N, rewards, eps = _new_q(), np.zeros((NS_MAX + 1, EW_MAX + 1, 2)), [], hp.epsilon
    for ep in range(hp.episodes):
        s, _ = env.reset(seed=int(rng.integers(2**31)))
        episode, done = [], False
        while not done:
            a = _eps_greedy(Q, s, eps, rng)
            s2, r, term, trunc, _ = env.step(a)
            episode.append((int(s[0]), int(s[1]), a, r))
            s, done = s2, term or trunc
        # first-visit returns, computed backwards
        G, first_G = 0.0, {}
        for ns, ew, a, r in reversed(episode):
            G = r + hp.gamma * G
            first_G[(ns, ew, a)] = G          # overwritten until the earliest visit remains
        for (ns, ew, a), g in first_G.items():
            N[ns, ew, a] += 1
            Q[ns, ew, a] += (g - Q[ns, ew, a]) / N[ns, ew, a]
        rewards.append(sum(x[3] for x in episode))
        eps = max(hp.epsilon_min, eps * hp.epsilon_decay)
    return Q, np.array(rewards)


def value_iteration(gamma: float = 0.9, tol: float = 1e-10, overflow_penalty: float = 0.0):
    """Exact optimal Q* for the known model (ground truth)."""
    P = transition_model(overflow_penalty)
    Q = _new_q()
    while True:
        V = Q.max(axis=2)
        Q_new = _new_q()
        for (ns, ew, a), outs in P.items():
            Q_new[ns, ew, a] = sum(p * (r + gamma * V[s2]) for p, s2, r in outs)
        if np.abs(Q_new - Q).max() < tol:
            return Q_new
        Q = Q_new


# ----------------------------------------------------------------- policies
def greedy_policy(Q):
    """Deterministic policy table pi[ns, ew] in {0: EW green, 1: NS green}."""
    return Q.argmax(axis=2)


def policy_from_table(pi):
    return lambda s, t: int(pi[s[0], s[1]])


def random_policy(rng):
    return lambda s, t: int(rng.integers(0, 2))


def fixed_time_policy(phase: int = 2):
    """Classic fixed-cycle signal: switch the green every `phase` steps."""
    return lambda s, t: EW_GREEN if (t // phase) % 2 == 0 else NS_GREEN


def longest_queue_policy():
    """Actuated rule of thumb: give green to the longer queue (ties -> EW, the larger road)."""
    return lambda s, t: NS_GREEN if s[0] > s[1] else EW_GREEN


def evaluate(policy, episodes: int = 2000, seed: int = 12345, metric: str = "waiting"):
    """Per-step average of ``info[metric]`` ("waiting" cars or "dropped" cars), with a 95% CI.

    Every policy sees the same random traffic (common random numbers), so
    differences between policies are not luck of the draw.
    """
    env = TrafficIntersectionEnv()
    per_ep = []
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        vals, done, t = [], False, 0
        while not done:
            s, r, term, trunc, info = env.step(policy(s, t))
            vals.append(info[metric])
            t, done = t + 1, term or trunc
        per_ep.append(np.mean(vals))
    per_ep = np.array(per_ep)
    ci = 1.96 * per_ep.std(ddof=1) / np.sqrt(len(per_ep))
    return float(per_ep.mean()), float(ci)
