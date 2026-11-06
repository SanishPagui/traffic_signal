"""Evaluate every controller on the held-out TEST set and write results/.

Test set = 5 hand-designed scenarios + held-out random demand, 5 seeds each
(30 episodes).  Every controller sees the identical vehicle files.

Outputs:
  results/test_episodes.csv     one row per (controller, episode)
  results/summary.md            tables + paired bootstrap CIs
  results/wait_by_scenario.png  small multiples per scenario
  results/training_curves.png   validation wait during training, per seed
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traffic_rl import config as C
from traffic_rl import controllers as K
from traffic_rl import demand
from traffic_rl.dqn import DQNAgent
from traffic_rl.env import SumoEnv

NAIVE = (30, 15, 30, 15)
BLUE, GRAY, INK, INK2, GRID = "#2a78d6", "#b5b3ad", "#0b0b0b", "#52514e", "#e6e5e1"
SEED_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]


def bootstrap_ci(x, n=10_000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n, len(x))).mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    tuned = tuple(json.load(open(os.path.join(C.RESULTS_DIR, "fixed_tuned.json")))["greens"])
    episodes = demand.test_set()
    env = SumoEnv()

    controllers = {
        f"Fixed naive {NAIVE[0]}/{NAIVE[1]}s": lambda r: K.run_fixed(env, r, NAIVE),
        f"Fixed tuned {tuned[0]}/{tuned[1]}s": lambda r: K.run_fixed(env, r, tuned),
        "Actuated (SUMO)": lambda r: K.run_actuated(env, r),
        "Max-pressure": lambda r: K.run_max_pressure(env, r),
    }
    seed_dirs = sorted(glob.glob(os.path.join(C.RUNS_DIR, "seed*")))
    for d in seed_dirs:
        agent = DQNAgent(env.obs_dim, env.n_actions)
        agent.load(os.path.join(d, "best.pt"))
        controllers[f"DQN {os.path.basename(d)}"] = (lambda a: lambda r: K.run_policy(env, r, a.policy()))(agent)

    rows = []
    for name, run in controllers.items():
        for i, (scen, route) in enumerate(episodes):
            m = run(route)
            rows.append({"controller": name, "scenario": scen, "episode": i, **m})
        w = np.mean([r["avg_wait"] for r in rows if r["controller"] == name])
        print(f"{name:24s} mean avg wait {w:7.2f} s", flush=True)

    with open(os.path.join(C.RESULTS_DIR, "test_episodes.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    # ---------------------------------------------------------------- tables
    names = list(controllers)
    dqn_names = [n for n in names if n.startswith("DQN")]
    base_names = [n for n in names if not n.startswith("DQN")]
    scens = list(dict.fromkeys(s for s, _ in episodes))

    def W(name, scen=None, key="avg_wait"):
        return np.array([r[key] for r in rows if r["controller"] == name
                         and (scen is None or r["scenario"] == scen)])

    def dqn_mean(scen=None, key="avg_wait"):  # average over seeds, per episode
        return np.mean([W(n, scen, key) for n in dqn_names], axis=0)

    lines = ["# Test results (held-out scenarios)", "",
             f"{len(episodes)} test episodes ({len(scens)} scenarios x 5 demand seeds). "
             "Metric: average waiting time per vehicle in seconds, counting time stopped "
             "(speed < 0.1 m/s) plus time queued before entering the network. Lower is better.", "",
             "## Average wait per vehicle (s) by scenario", "",
             "| Scenario | " + " | ".join(base_names) + " | DQN (mean of %d seeds) |" % len(dqn_names),
             "|---" * (len(base_names) + 2) + "|"]
    for s in scens + [None]:
        vals = [f"{W(n, s).mean():.1f}" for n in base_names]
        seed_means = [W(n, s).mean() for n in dqn_names]
        lines.append(f"| {s or '**All**'} | " + " | ".join(vals) +
                     f" | **{np.mean(seed_means):.1f}** ± {np.std(seed_means):.1f} |")
    lines += ["", "± = std across training seeds.", "",
              "## Paired improvement of DQN over each baseline (all test episodes)", "",
              "| Baseline | Baseline wait | DQN wait | Reduction | 95% bootstrap CI | DQN better in |",
              "|---|---|---|---|---|---|"]
    d = dqn_mean()
    for b in base_names:
        bw = W(b)
        rel = (bw - d) / bw * 100
        lo, hi = bootstrap_ci(rel)
        lines.append(f"| {b} | {bw.mean():.1f} s | {d.mean():.1f} s | {rel.mean():.1f}% | "
                     f"[{lo:.1f}%, {hi:.1f}%] | {(d < bw).sum()}/{len(d)} episodes |")
    lines += ["", "## Per-seed DQN results vs tuned fixed timer", "",
              "| Seed | Avg wait (s) | Reduction vs tuned fixed | Beats tuned fixed in |",
              "|---|---|---|---|"]
    tw = W(base_names[1])
    for n in dqn_names:
        lines.append(f"| {n} | {W(n).mean():.1f} | {(tw - W(n)).sum() / tw.sum() * 100:.1f}% | "
                     f"{(W(n) < tw).sum()}/{len(tw)} episodes |")
    lines += ["", "## Efficiency, fairness and safety (all 30 test episodes)", "",
              "Time loss = extra travel time compared with free flow. p95 / max = per-vehicle wait "
              "(mean over episodes of the episode's 95th percentile / worst vehicle). Guard = times "
              f"the {C.MAX_RED}s max-red safety rule overrode the controller (actuated control is "
              "bounded by its own maxDur and does not use the guard).", "",
              "| Controller | Time loss (s) | p95 wait (s) | Worst vehicle wait (s) | Unfinished vehicles | Guard overrides |",
              "|---|---|---|---|---|---|"]
    for n in names:
        lines.append(f"| {n} | {W(n, key='avg_time_loss').mean():.1f} | {W(n, key='p95_wait').mean():.1f} | "
                     f"{W(n, key='max_wait').mean():.1f} | {int(W(n, key='unfinished').sum())} | "
                     f"{int(W(n, key='guard_overrides').sum())} |")
    with open(os.path.join(C.RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    plot_scenarios(scens, base_names, W, dqn_names)
    plot_training(tuned)


def _style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def plot_scenarios(scens, base_names, W, dqn_names):
    labels = base_names + ["DQN (3-seed mean)"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=True)
    for ax, s in zip(axes.flat, scens + [None]):
        vals = [W(n, s).mean() for n in base_names] + [np.mean([W(n, s).mean() for n in dqn_names])]
        y = np.arange(len(labels))[::-1]
        ax.barh(y, vals, height=0.6, color=[GRAY] * len(base_names) + [BLUE], edgecolor="white", lw=2)
        for yi, v in zip(y, vals):
            ax.text(v, yi, f" {v:.1f}", va="center", fontsize=9, color=INK)
        ax.set_yticks(y, labels, fontsize=9, color=INK)
        ax.set_title(s or "All test episodes", loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlim(0, max(vals) * 1.25)
        _style(ax)
    fig.suptitle("Average wait per vehicle (s), held-out test scenarios — lower is better",
                 x=0.01, ha="left", fontsize=13, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS_DIR, "wait_by_scenario.png"), dpi=150, facecolor="white")
    plt.close(fig)


def plot_training(tuned):
    env = SumoEnv()
    val = demand.val_set()
    refs = {f"Fixed tuned {tuned[0]}/{tuned[1]}s": np.mean([K.run_fixed(env, r, tuned)["avg_wait"] for r in val]),
            "Max-pressure": np.mean([K.run_max_pressure(env, r)["avg_wait"] for r in val]),
            "Actuated": np.mean([K.run_actuated(env, r)["avg_wait"] for r in val])}
    fig, ax = plt.subplots(figsize=(10, 5))
    ymax = max(refs.values()) * 1.6
    for i, d in enumerate(sorted(glob.glob(os.path.join(C.RUNS_DIR, "seed*")))):
        log = [r for r in json.load(open(os.path.join(d, "log.json"))) if "val_wait" in r]
        x = [r["episode"] + 1 for r in log]
        y = [min(r["val_wait"], ymax) for r in log]
        ax.plot(x, y, color=SEED_COLORS[i % 3], lw=2, marker="o", ms=4, label=f"DQN {os.path.basename(d)}")
    for name, v in refs.items():
        ax.axhline(v, color=INK2, lw=1, ls="--")
        ax.text(ax.get_xlim()[1], v, f" {name} ({v:.1f})", va="center", fontsize=9, color=INK2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Training episode", color=INK2)
    ax.set_ylabel("Validation avg wait (s), clipped", color=INK2)
    ax.set_title("Greedy-policy validation wait during training", loc="left", color=INK, fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    _style(ax)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(os.path.join(C.RESULTS_DIR, "training_curves.png"), dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
