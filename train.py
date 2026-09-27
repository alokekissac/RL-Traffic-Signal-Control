"""
Train and evaluate every agent, then write models/, results/ and docs/figures/.

    python train.py              # full run (5 seeds, ~a few minutes on a laptop)
    python train.py --quick      # smoke test

Experiments
-----------
  baseline : the report's main settings   alpha .1  gamma .9   eps 1.0  decay .995  min .01  5000 episodes
  exp1     : report experiment 1          alpha .2  gamma .3   eps 1.0  decay .456  min .03   500 episodes
  exp2     : report experiment 2          alpha .3  gamma .45  eps 0.5  decay .564  min .02   500 episodes
  overflow : extension - baseline settings, but each dropped car costs 2 points

Each (experiment, algorithm) pair is trained with several independent seeds.
Policies are evaluated on 2,000 held-out episodes of identical random traffic
(common random numbers), and scored by the average number of cars waiting.
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from traffic_rl.agents import (Hyperparams, evaluate, fixed_time_policy, greedy_policy,
                               longest_queue_policy, monte_carlo, policy_from_table,
                               q_learning, random_policy, value_iteration)

ROOT = Path(__file__).resolve().parent
EXPERIMENTS = {
    "baseline": Hyperparams(0.1, 0.9, 1.0, 0.995, 0.01, 5000),
    "exp1": Hyperparams(0.2, 0.3, 1.0, 0.456, 0.03, 500),
    "exp2": Hyperparams(0.3, 0.45, 0.5, 0.564, 0.02, 500),
    "overflow": Hyperparams(0.1, 0.9, 1.0, 0.995, 0.01, 5000, overflow_penalty=2.0),
}
ALGOS = {"q_learning": q_learning, "monte_carlo": monte_carlo}


def _run(job):
    exp, algo, seed, hp = job
    Q, rewards = ALGOS[algo](hp, seed=seed)
    pi = greedy_policy(Q)
    score, ci = evaluate(policy_from_table(pi))
    dropped, _ = evaluate(policy_from_table(pi), metric="dropped")
    return exp, algo, seed, Q, rewards, score, ci, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        for hp in EXPERIMENTS.values():
            hp.episodes = min(hp.episodes, 300)
        args.seeds = 2

    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)

    # ---- ground truth
    Q_star = value_iteration(gamma=0.9)
    pi_star = greedy_policy(Q_star)
    Q_star_ov = value_iteration(gamma=0.9, overflow_penalty=2.0)
    pi_star_ov = greedy_policy(Q_star_ov)

    # ---- learning agents
    jobs = [(e, a, s, hp) for e, hp in EXPERIMENTS.items() for a in ALGOS for s in range(args.seeds)]
    runs = {}
    with ProcessPoolExecutor() as pool:
        for exp, algo, seed, Q, rewards, score, ci, dropped in pool.map(_run, jobs):
            runs.setdefault((exp, algo), []).append(dict(seed=seed, Q=Q, rewards=rewards, score=score,
                                                         ci=ci, dropped=dropped))
            print(f"{exp:9s} {algo:12s} seed {seed}: {score:.3f} cars waiting")

    # ---- baselines
    base_pols = {
        "Random": lambda: random_policy(np.random.default_rng(7)),
        "Fixed-time (2-step cycle)": lambda: fixed_time_policy(2),
        "Longest queue first": longest_queue_policy,
        "Always EW green": lambda: (lambda s, t: 0),
        "Optimal (value iteration)": lambda: policy_from_table(pi_star),
        "Optimal, overflow-aware": lambda: policy_from_table(pi_star_ov),
    }
    base = {k: (*evaluate(f()), evaluate(f(), metric="dropped")[0]) for k, f in base_pols.items()}

    # ---- summarise
    summary = {"baselines": {k: {"mean": v[0], "ci95": v[1], "dropped": v[2]} for k, v in base.items()},
               "experiments": {}}
    for (exp, algo), rs in runs.items():
        scores = np.array([r["score"] for r in rs])
        ref = pi_star_ov if exp == "overflow" else pi_star
        agree = [float((greedy_policy(r["Q"]) == ref).mean()) for r in rs]
        last = [float(r["rewards"][-100:].mean()) for r in rs]
        summary["experiments"].setdefault(exp, {"hyperparams": vars(EXPERIMENTS[exp])})[algo] = {
            "mean_waiting": float(scores.mean()),
            "std_across_seeds": float(scores.std(ddof=1)) if len(scores) > 1 else 0.0,
            "best_seed_waiting": float(scores.min()),
            "mean_dropped": float(np.mean([r["dropped"] for r in rs])),
            "policy_agreement_with_optimal": float(np.mean(agree)),
            "final_100_episode_return": float(np.mean(last)),
            "per_seed": [float(x) for x in scores],
        }
    (ROOT / "results" / "results.json").write_text(json.dumps(summary, indent=2))

    # ---- save models (best seed of the baseline experiment) + optimal
    def dump(name, Q, meta):
        (ROOT / "models" / f"{name}.json").write_text(json.dumps(
            {**meta, "state": "Q[ns][ew] = [Q(EW green), Q(NS green)]",
             "Q": np.round(Q, 4).tolist()}, indent=1))
    for algo in ALGOS:
        best = min(runs[("baseline", algo)], key=lambda r: r["score"])
        dump(algo, best["Q"], {"algorithm": algo, "seed": best["seed"],
                               "hyperparams": vars(EXPERIMENTS["baseline"]),
                               "eval_mean_waiting": best["score"]})
        np.save(ROOT / "results" / f"curves_{algo}.npy",
                np.stack([r["rewards"] for r in runs[("baseline", algo)]]))
    dump("optimal", Q_star, {"algorithm": "value_iteration", "gamma": 0.9})
    dump("optimal_overflow_aware", Q_star_ov, {"algorithm": "value_iteration", "gamma": 0.9,
                                               "overflow_penalty": 2.0})
    best = min(runs[("overflow", "q_learning")], key=lambda r: r["score"] + 2 * r["dropped"])
    dump("q_learning_overflow_aware", best["Q"], {"algorithm": "q_learning", "seed": best["seed"],
                                                  "hyperparams": vars(EXPERIMENTS["overflow"])})

    # ---- markdown table
    lines = ["| Policy | Avg cars waiting ↓ | 95% CI | Cars turned away / step | Matches optimal policy |",
             "|---|---:|---:|---:|---:|"]
    rows = [(k, v[0], v[1], v[2], "–") for k, v in base.items()]
    for algo, label in (("q_learning", "Q-learning"), ("monte_carlo", "Monte Carlo (first-visit)")):
        e = summary["experiments"]["baseline"][algo]
        rows.append((f"{label}, mean of {args.seeds} seeds", e["mean_waiting"], None, e["mean_dropped"],
                     f"{e['policy_agreement_with_optimal']:.0%}"))
    for name, m, ci, dr, ag in sorted(rows, key=lambda r: r[1]):
        lines.append(f"| {name} | {m:.3f} | {'±%.3f' % ci if ci else '–'} | {dr:.3f} | {ag} |")
    lines += ["", "| Experiment | Algorithm | α | γ | ε₀ | decay | ε_min | Episodes | Avg cars waiting ↓ (± sd over seeds) | Turned away / step | Matches optimal |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for exp, d in summary["experiments"].items():
        hp = d["hyperparams"]
        for algo in ALGOS:
            e = d[algo]
            lines.append(f"| {exp} | {algo} | {hp['alpha']} | {hp['gamma']} | {hp['epsilon']} | {hp['epsilon_decay']} "
                         f"| {hp['epsilon_min']} | {hp['episodes']} | {e['mean_waiting']:.3f} ± {e['std_across_seeds']:.3f} "
                         f"| {e['mean_dropped']:.3f} | {e['policy_agreement_with_optimal']:.0%} |")
    (ROOT / "results" / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    from make_figures import make_all
    make_all(ROOT)


if __name__ == "__main__":
    main()
