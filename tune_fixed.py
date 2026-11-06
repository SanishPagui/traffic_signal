"""Grid-search the fixed-time plan on the VALIDATION scenarios so the
fixed-timer baseline is as strong as a traffic engineer would make it.
Writes results/fixed_tuned.json."""
import itertools
import json
import os

import numpy as np

from traffic_rl import config as C
from traffic_rl import demand
from traffic_rl.controllers import run_fixed
from traffic_rl.env import SumoEnv


def main():
    env = SumoEnv()
    routes = demand.val_set()
    best = None
    for through, left in itertools.product(range(10, 51, 5), range(5, 31, 5)):
        g = (through, left, through, left)
        w = float(np.mean([run_fixed(env, r, g)["avg_wait"] for r in routes]))
        print(f"through {through:2d}s left {left:2d}s -> val wait {w:6.2f}", flush=True)
        if best is None or w < best[1]:
            best = (g, w)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "fixed_tuned.json"), "w") as f:
        json.dump({"greens": best[0], "val_wait": best[1]}, f, indent=2)
    print("best plan:", best)


if __name__ == "__main__":
    main()
