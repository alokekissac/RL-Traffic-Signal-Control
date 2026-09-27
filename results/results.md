| Policy | Avg cars waiting ↓ | 95% CI | Cars turned away / step | Matches optimal policy |
|---|---:|---:|---:|---:|
| Optimal (value iteration) | 3.561 | ±0.017 | 0.961 | – |
| Always EW green | 3.564 | ±0.016 | 0.974 | – |
| Q-learning, mean of 5 seeds | 3.570 | – | 0.972 | 89% |
| Monte Carlo (first-visit), mean of 5 seeds | 3.699 | – | 0.813 | 86% |
| Optimal, overflow-aware | 3.833 | ±0.022 | 0.572 | – |
| Longest queue first | 5.070 | ±0.028 | 0.491 | – |
| Random | 5.496 | ±0.030 | 0.561 | – |
| Fixed-time (2-step cycle) | 5.708 | ±0.029 | 0.510 | – |

| Experiment | Algorithm | α | γ | ε₀ | decay | ε_min | Episodes | Avg cars waiting ↓ (± sd over seeds) | Turned away / step | Matches optimal |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | q_learning | 0.1 | 0.9 | 1.0 | 0.995 | 0.01 | 5000 | 3.570 ± 0.009 | 0.972 | 89% |
| baseline | monte_carlo | 0.1 | 0.9 | 1.0 | 0.995 | 0.01 | 5000 | 3.699 ± 0.173 | 0.813 | 86% |
| exp1 | q_learning | 0.2 | 0.3 | 1.0 | 0.456 | 0.03 | 500 | 4.061 ± 0.448 | 0.946 | 65% |
| exp1 | monte_carlo | 0.2 | 0.3 | 1.0 | 0.456 | 0.03 | 500 | 3.993 ± 0.242 | 0.964 | 69% |
| exp2 | q_learning | 0.3 | 0.45 | 0.5 | 0.564 | 0.02 | 500 | 3.874 ± 0.281 | 0.917 | 69% |
| exp2 | monte_carlo | 0.3 | 0.45 | 0.5 | 0.564 | 0.02 | 500 | 4.153 ± 0.118 | 0.966 | 60% |
| overflow | q_learning | 0.1 | 0.9 | 1.0 | 0.995 | 0.01 | 5000 | 3.873 ± 0.041 | 0.577 | 91% |
| overflow | monte_carlo | 0.1 | 0.9 | 1.0 | 0.995 | 0.01 | 5000 | 3.961 ± 0.296 | 0.637 | 84% |
