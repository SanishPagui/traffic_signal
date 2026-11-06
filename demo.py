"""Live demo in sumo-gui.

Side-by-side (recommended for presenting): fixed timer on the left, AI on the right,
same cars, same moment, live scoreboard in this terminal.

    python demo.py --compare
    python demo.py --compare --scenario balanced_high --speed 10

Single controller:

    python demo.py --controller dqn --scenario shifting

Stopped cars are drawn RED, moving cars WHITE, so queues are easy to see.
"""
import argparse
import json
import os
import subprocess
import sys

from traffic_rl import config as C
from traffic_rl import controllers as K
from traffic_rl import demand
from traffic_rl.dqn import DQNAgent
from traffic_rl.env import SumoEnv

LABELS = {"dqn": "AI (DQN)", "fixed": "Fixed timer", "actuated": "Actuated", "maxpressure": "Max-pressure"}
RED, WHITE = (230, 40, 40, 255), (240, 240, 240, 255)


def make_hook(label, report_every):
    def hook(env):
        tr = env.traci
        for v in tr.vehicle.getIDList():
            stopped = tr.vehicle.getSpeed(v) < 0.1
            if stopped != hook.state.get(v):
                tr.vehicle.setColor(v, RED if stopped else WHITE)
                hook.state[v] = stopped
        hook.departed += tr.simulation.getDepartedNumber()
        if env.t % report_every == 0:
            queued = int(sum(env.lane_halting()))
            avg = env.total_cost / max(hook.departed, 1)
            print(f"{label:12s} | clock {env.t // 60:3d} min | cars stopped now {queued:3d} "
                  f"| avg wait so far {avg:5.1f} s", flush=True)
    hook.state, hook.departed = {}, 0
    return hook


def run_one(args):
    label = LABELS[args.controller]
    gui_args = ["--delay", str(args.delay), "--window-size", "940,820"]
    if args.window_pos:
        gui_args += ["--window-pos", args.window_pos]
    env = SumoEnv(gui=True, gui_args=gui_args, hook=make_hook(label, 300))
    route = demand.make_named(args.scenario, C.TEST_SEED_BASE)
    if args.controller == "dqn":
        agent = DQNAgent(env.obs_dim, env.n_actions)
        agent.load(os.path.join(C.RUNS_DIR, f"seed{args.seed}", "best.pt"))
        m = K.run_policy(env, route, agent.policy())
    elif args.controller == "fixed":
        greens = json.load(open(os.path.join(C.RESULTS_DIR, "fixed_tuned.json")))["greens"]
        m = K.run_fixed(env, route, tuple(greens))
    elif args.controller == "actuated":
        m = K.run_actuated(env, route)
    else:
        m = K.run_max_pressure(env, route)
    print(f"\n=== {label} FINAL ({args.scenario}): average wait {m['avg_wait']:.1f} s per car, "
          f"95% of cars waited under {m['p95_wait']:.0f} s, {m['vehicles']} cars ===\n", flush=True)


def run_compare(args):
    """Launch two processes, one GUI window each, running at the same speed."""
    procs = []
    for ctrl, pos in [(args.left, "0,0"), (args.right, "950,0")]:
        procs.append(subprocess.Popen([sys.executable, __file__, "--controller", ctrl,
                                       "--scenario", args.scenario, "--seed", str(args.seed),
                                       "--delay", str(args.delay), "--window-pos", pos]))
    for p in procs:
        p.wait()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare", action="store_true", help="side-by-side: --left vs --right")
    ap.add_argument("--left", default="fixed", choices=list(LABELS))
    ap.add_argument("--right", default="dqn", choices=list(LABELS))
    ap.add_argument("--controller", choices=list(LABELS), default="dqn")
    ap.add_argument("--scenario", choices=demand.NAMED, default="shifting")
    ap.add_argument("--seed", type=int, default=0, help="DQN training seed to load")
    ap.add_argument("--delay", type=int, default=15, help="ms per simulated second (lower = faster)")
    ap.add_argument("--speed", type=int, help="shortcut: speed-up factor, e.g. 10 = 10x real time")
    ap.add_argument("--window-pos", default=None)
    args = ap.parse_args()
    if args.speed:
        args.delay = max(1000 // args.speed, 0)
    run_compare(args) if args.compare else run_one(args)
