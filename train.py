"""Train a DQN signal controller.

Each episode samples a fresh random demand scenario (train seeds only).
Every `--val-every` episodes the greedy policy is evaluated on held-out
validation scenarios; the checkpoint with the lowest validation wait is kept.
Test scenarios are never touched here.

    python train.py --seed 0 --episodes 300
"""
import argparse
import json
import os
import time

import numpy as np

from traffic_rl import config as C
from traffic_rl import demand
from traffic_rl.controllers import run_policy
from traffic_rl.dqn import DQNAgent
from traffic_rl.env import SumoEnv

REWARD_CLIP = 10.0  # scaled reward per step; ~200 halted vehicles for 5 s


def validate(env, agent, routes):
    return float(np.mean([run_policy(env, r, agent.policy())["avg_wait"] for r in routes]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--val-every", type=int, default=10)
    ap.add_argument("--eps-frac", type=float, default=0.4, help="fraction of episodes to anneal epsilon")
    args = ap.parse_args()

    out = os.path.join(C.RUNS_DIR, f"seed{args.seed}")
    os.makedirs(out, exist_ok=True)
    env = SumoEnv(sumo_seed=args.seed, max_time=C.DEMAND_END + 900)
    val_env = SumoEnv(sumo_seed=args.seed)  # full-length cap, same as test
    agent = DQNAgent(env.obs_dim, env.n_actions, seed=args.seed)
    val_routes = demand.val_set()

    log, best = [], float("inf")
    for ep in range(args.episodes):
        eps = max(0.05, 1.0 - ep / (args.eps_frac * args.episodes))
        route = demand.make_random(C.TRAIN_SEED_BASE + args.seed * 10_000 + ep, tag="train")
        t0 = time.time()
        obs, done, ret, losses = env.reset(route), False, 0.0, []
        while not done:
            a = agent.act(obs, eps)
            obs2, r, done = env.step(a)
            # only a drained network is terminal; hitting the time cap is truncation
            # store the action actually executed (the safety guard may override it)
            agent.buf.add(obs, env.last_action, max(r, -REWARD_CLIP), obs2, float(env.drained()))
            loss = agent.update()
            if loss is not None:
                losses.append(loss)
            obs, ret = obs2, ret + r
        m = env.metrics()
        row = {"episode": ep, "eps": eps, "return": ret, "train_wait": m["avg_wait"],
               "loss": float(np.mean(losses)) if losses else None, "secs": time.time() - t0}

        if (ep + 1) % args.val_every == 0:
            v = validate(val_env, agent, val_routes)
            row["val_wait"] = v
            if v < best:
                best = v
                agent.save(os.path.join(out, "best.pt"))
            agent.save(os.path.join(out, "last.pt"))
            print(f"[seed {args.seed}] ep {ep+1:4d} eps {eps:.2f} train_wait {m['avg_wait']:6.2f} "
                  f"val_wait {v:6.2f} best {best:6.2f}", flush=True)
        log.append(row)
        with open(os.path.join(out, "log.json"), "w") as f:
            json.dump(log, f)


if __name__ == "__main__":
    main()
