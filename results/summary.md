# Test results (held-out scenarios)

30 test episodes (6 scenarios x 5 demand seeds). Metric: average waiting time per vehicle in seconds, counting time stopped (speed < 0.1 m/s) plus time queued before entering the network. Lower is better.

## Average wait per vehicle (s) by scenario

| Scenario | Fixed naive 30/15s | Fixed tuned 20/15s | Actuated (SUMO) | Max-pressure | DQN (mean of 3 seeds) |
|---|---|---|---|---|---|
| balanced_low | 26.4 | 22.9 | 10.7 | 9.3 | **9.6** ± 0.3 |
| balanced_high | 28.8 | 24.7 | 17.1 | 26.7 | **15.5** ± 0.2 |
| ns_major | 29.4 | 26.9 | 12.9 | 12.8 | **12.3** ± 0.7 |
| shifting | 28.6 | 25.1 | 12.7 | 12.2 | **11.8** ± 0.1 |
| tidal | 28.1 | 25.6 | 13.9 | 18.4 | **13.7** ± 0.3 |
| random_heldout | 32.1 | 29.4 | 16.6 | 39.5 | **14.5** ± 0.1 |
| **All** | 28.9 | 25.8 | 14.0 | 19.8 | **12.9** ± 0.3 |

± = std across training seeds.

## Paired improvement of DQN over each baseline (all test episodes)

| Baseline | Baseline wait | DQN wait | Reduction | 95% bootstrap CI | DQN better in |
|---|---|---|---|---|---|
| Fixed naive 30/15s | 28.9 s | 12.9 s | 55.6% | [53.4%, 57.9%] | 30/30 episodes |
| Fixed tuned 20/15s | 25.8 s | 12.9 s | 49.9% | [47.3%, 52.4%] | 30/30 episodes |
| Actuated (SUMO) | 14.0 s | 12.9 s | 6.8% | [4.9%, 8.7%] | 29/30 episodes |
| Max-pressure | 19.8 s | 12.9 s | 14.3% | [6.9%, 22.4%] | 24/30 episodes |

## Per-seed DQN results vs tuned fixed timer

| Seed | Avg wait (s) | Reduction vs tuned fixed | Beats tuned fixed in |
|---|---|---|---|
| DQN seed0 | 12.8 | 50.3% | 30/30 episodes |
| DQN seed1 | 13.3 | 48.4% | 30/30 episodes |
| DQN seed2 | 12.7 | 50.8% | 30/30 episodes |

## Efficiency, fairness and safety (all 30 test episodes)

Time loss = extra travel time compared with free flow. p95 / max = per-vehicle wait (mean over episodes of the episode's 95th percentile / worst vehicle). Guard = times the 120s max-red safety rule overrode the controller (actuated control is bounded by its own maxDur and does not use the guard).

| Controller | Time loss (s) | p95 wait (s) | Worst vehicle wait (s) | Unfinished vehicles | Guard overrides |
|---|---|---|---|---|---|
| Fixed naive 30/15s | 37.1 | 72.5 | 119.4 | 0 | 0 |
| Fixed tuned 20/15s | 34.2 | 60.0 | 90.9 | 0 | 0 |
| Actuated (SUMO) | 21.5 | 38.2 | 80.0 | 0 | 0 |
| Max-pressure | 30.4 | 84.4 | 241.6 | 0 | 98 |
| DQN seed0 | 20.5 | 41.7 | 123.9 | 0 | 26 |
| DQN seed1 | 21.0 | 44.2 | 130.7 | 0 | 21 |
| DQN seed2 | 20.3 | 40.4 | 134.9 | 0 | 43 |
