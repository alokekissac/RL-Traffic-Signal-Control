"""
Gymnasium environment: a single intersection with two approaches.

Dynamics (identical to the original CA2 notebook):
  1. Arrivals:   0-2 cars join each queue (uniform).
  2. Green phase: the green approach discharges 1-2 cars (uniform), never below 0.
  3. Capacity:   queues are capped at NS_MAX (3) and EW_MAX (5) cars.
  4. Reward:     -(cars waiting on both approaches after the step).

Cars that arrive at a full queue are dropped (they "vanish"). The original
reward ignores them, which lets an agent starve the short North-South road.
Set ``overflow_penalty > 0`` to charge that many points per dropped car
(an extension beyond the original coursework; the default of 0 reproduces it).

State  : (ns, ew) queue lengths  -> MultiDiscrete([4, 6])  (24 states)
Action : 0 = East-West green, 1 = North-South green
Episode: 50 decisions (truncated), random initial queues.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

NS_MAX = 3
EW_MAX = 5
EW_GREEN, NS_GREEN = 0, 1
ARRIVALS = (0, 1, 2)       # cars arriving per approach per step
DEPARTURES = (1, 2)        # cars discharged by the green approach per step


class TrafficIntersectionEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, episode_length: int = 50, overflow_penalty: float = 0.0,
                 render_mode: str | None = None):
        super().__init__()
        self.episode_length = episode_length
        self.overflow_penalty = overflow_penalty
        self.render_mode = render_mode
        self.observation_space = spaces.MultiDiscrete([NS_MAX + 1, EW_MAX + 1])
        self.action_space = spaces.Discrete(2)
        self._state = (0, 0)
        self._t = 0

    # --------------------------------------------------------------- gym api
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "state" in options:
            self._state = tuple(int(x) for x in options["state"])
        else:
            self._state = (int(self.np_random.integers(0, NS_MAX + 1)),
                           int(self.np_random.integers(0, EW_MAX + 1)))
        self._t = 0
        return np.array(self._state, dtype=np.int64), {}

    def step(self, action: int):
        ns, ew = self._state
        rng = self.np_random
        ns += int(rng.choice(ARRIVALS))
        ew += int(rng.choice(ARRIVALS))
        if action == EW_GREEN:
            ew = max(0, ew - int(rng.choice(DEPARTURES)))
        else:
            ns = max(0, ns - int(rng.choice(DEPARTURES)))
        dropped = max(0, ns - NS_MAX) + max(0, ew - EW_MAX)
        ns, ew = min(ns, NS_MAX), min(ew, EW_MAX)
        self._state = (ns, ew)
        self._t += 1
        reward = -float(ns + ew) - self.overflow_penalty * dropped
        truncated = self._t >= self.episode_length
        info = {"waiting": ns + ew, "dropped": dropped}
        return np.array(self._state, dtype=np.int64), reward, False, truncated, info

    def render(self):
        ns, ew = self._state
        return f"t={self._t:02d}  NS {'█' * ns:<{NS_MAX}} {ns}   EW {'█' * ew:<{EW_MAX}} {ew}"


def transition_model(overflow_penalty: float = 0.0):
    """Exact transition probabilities P[s][a] -> list of (prob, next_state, reward).

    The state space is tiny (24 states), so the MDP can be solved exactly with
    dynamic programming - a ground truth for judging the learned policies.
    """
    P = {}
    for ns in range(NS_MAX + 1):
        for ew in range(EW_MAX + 1):
            for a in (EW_GREEN, NS_GREEN):
                outcomes: dict[tuple[tuple[int, int], int], float] = {}
                p = 1.0 / (len(ARRIVALS) ** 2 * len(DEPARTURES))
                for an in ARRIVALS:
                    for ae in ARRIVALS:
                        for d in DEPARTURES:
                            n2, e2 = ns + an, ew + ae
                            if a == EW_GREEN:
                                e2 = max(0, e2 - d)
                            else:
                                n2 = max(0, n2 - d)
                            drop = max(0, n2 - NS_MAX) + max(0, e2 - EW_MAX)
                            key = ((min(n2, NS_MAX), min(e2, EW_MAX)), drop)
                            outcomes[key] = outcomes.get(key, 0.0) + p
                P[(ns, ew, a)] = [(pr, s2, -float(sum(s2)) - overflow_penalty * drop)
                                  for (s2, drop), pr in outcomes.items()]
    return P
