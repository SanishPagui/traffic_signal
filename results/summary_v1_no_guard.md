# Test results (held-out scenarios)

30 test episodes (6 scenarios x 5 demand seeds). Metric: average waiting time per vehicle in seconds, counting time stopped (speed < 0.1 m/s) plus time queued before entering the network. Lower is better.

## Average wait per vehicle (s) by scenario

| Scenario | Fixed naive 30/15s | Fixed tuned 20/15s | Actuated (SUMO) | Max-pressure | DQN (mean of 3 seeds) |
|---|---|---|---|---|---|
| balanced_low | 26.4 | 22.9 | 10.7 | 9.3 | **9.0** ± 0.2 |
| balanced_high | 28.8 | 24.7 | 17.1 | 26.7 | **15.6** ± 0.5 |
| ns_major | 29.4 | 26.9 | 12.9 | 13.1 | **12.5** ± 0.5 |
| shifting | 28.6 | 25.1 | 12.7 | 11.9 | **11.3** ± 0.3 |
| tidal | 28.1 | 25.6 | 13.9 | 19.1 | **13.8** ± 0.6 |
| random_heldout | 32.1 | 29.4 | 16.6 | 40.3 | **15.2** ± 1.7 |
| **All** | 28.9 | 25.8 | 14.0 | 20.1 | **12.9** ± 0.3 |

± = std across training seeds.

## Paired improvement of DQN over each baseline (all test episodes)

| Baseline | Baseline wait | DQN wait | Reduction | 95% bootstrap CI | DQN better in |
|---|---|---|---|---|---|
| Fixed naive 30/15s | 28.9 s | 12.9 s | 55.9% | [53.0%, 58.7%] | 30/30 episodes |
| Fixed tuned 20/15s | 25.8 s | 12.9 s | 50.2% | [47.2%, 53.3%] | 30/30 episodes |
| Actuated (SUMO) | 14.0 s | 12.9 s | 7.6% | [4.9%, 10.3%] | 25/30 episodes |
| Max-pressure | 20.1 s | 12.9 s | 16.4% | [9.8%, 23.8%] | 26/30 episodes |

## Per-seed DQN results vs tuned fixed timer

| Seed | Avg wait (s) | Reduction vs tuned fixed | Avg time loss (s) | Unfinished vehicles |
|---|---|---|---|---|
| DQN seed0 | 12.5 | 51.5% | 20.0 | 0 |
| DQN seed1 | 13.2 | 48.6% | 20.8 | 5 |
| DQN seed2 | 13.0 | 49.5% | 20.7 | 11 |

## Average time loss per vehicle (s)

| Controller | Time loss |
|---|---|
| Fixed naive 30/15s | 37.1 |
| Fixed tuned 20/15s | 34.2 |
| Actuated (SUMO) | 21.5 |
| Max-pressure | 30.7 |
| DQN (mean of seeds) | 20.5 |
