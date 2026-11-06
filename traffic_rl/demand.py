"""Seeded, time-varying Poisson demand.

A scenario is a list of segments; each segment gives per-approach flow (veh/h)
for an interval of the first hour.  Every vehicle is sampled from an RNG seeded
by `seed`, so a (scenario, seed) pair is exactly reproducible.
"""
import os

import numpy as np

from . import config as C
from .network import DEST

SEG = 900  # s per demand segment (4 segments per hour)


def _turn_probs(rng=None):
    if rng is None:
        return {"r": 0.2, "s": 0.6, "l": 0.2}
    l = rng.uniform(0.1, 0.3)
    r = rng.uniform(0.1, 0.25)
    return {"r": r, "s": 1 - l - r, "l": l}


def named_scenario(name):
    """Hand-designed test scenarios. Returns list of 4 dicts approach->veh/h."""
    def seg(n, e, s, w):
        return {"N": n, "E": e, "S": s, "W": w}
    if name == "balanced_low":
        return [seg(400, 400, 400, 400)] * 4
    if name == "balanced_high":
        return [seg(700, 700, 700, 700)] * 4
    if name == "ns_major":
        return [seg(850, 300, 850, 300)] * 4
    if name == "shifting":  # NS peak moves to EW peak over the hour
        return [seg(850, 250, 850, 250), seg(650, 450, 650, 450),
                seg(450, 650, 450, 650), seg(250, 850, 250, 850)]
    if name == "tidal":     # one-directional surges (morning/evening style)
        return [seg(900, 300, 350, 600), seg(800, 350, 400, 550),
                seg(400, 550, 850, 350), seg(350, 600, 900, 300)]
    raise ValueError(name)


NAMED = ["balanced_low", "balanced_high", "ns_major", "shifting", "tidal"]


def random_scenario(rng):
    """Random demand for training/validation: varying level, asymmetry and drift."""
    level = rng.uniform(250, 800)
    segs = []
    w = rng.uniform(0.5, 1.5, size=4)
    for _ in range(C.DEMAND_END // SEG):
        w = np.clip(w * rng.uniform(0.7, 1.4, size=4), 0.3, 2.0)
        flows = np.clip(level * w, 100, 950)
        segs.append(dict(zip(C.APPROACHES, flows.tolist())))
    return segs


def write_routes(path, segments, seed, random_turns=False):
    rng = np.random.default_rng(seed)
    turns = {a: _turn_probs(rng if random_turns else None) for a in C.APPROACHES}
    lines = ['<routes>',
             '  <vType id="car" accel="2.6" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="13.89"/>']
    for o in C.APPROACHES:
        for t in "rsl":
            lines.append(f'  <route id="{o}_{t}" edges="{o}2C C2{DEST[(o, t)]}"/>')
    vid = 0
    for t in range(C.DEMAND_END):
        flows = segments[min(t // SEG, len(segments) - 1)]
        for o in C.APPROACHES:
            p = flows[o] / 3600.0
            if rng.random() < p:
                tp = turns[o]
                turn = rng.choice(["r", "s", "l"], p=[tp["r"], tp["s"], tp["l"]])
                lane = "2" if turn == "l" else ("best" if turn == "s" else "0")
                lines.append(f'  <vehicle id="v{vid}" type="car" route="{o}_{turn}" depart="{t}" '
                             f'departLane="{lane}" departSpeed="max"/>')
                vid += 1
    lines.append('</routes>')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def make_random(seed, tag="train"):
    rng = np.random.default_rng(seed + 7919)
    path = os.path.join(C.ROUTE_DIR, tag, f"random_{seed}.rou.xml")
    if not os.path.exists(path):
        write_routes(path, random_scenario(rng), seed, random_turns=True)
    return path


def make_named(name, seed):
    path = os.path.join(C.ROUTE_DIR, "test", f"{name}_{seed}.rou.xml")
    if not os.path.exists(path):
        write_routes(path, named_scenario(name), seed)
    return path


def test_set(n_seeds=5):
    """(label, route_file) pairs: 5 named scenarios + held-out random, n_seeds each."""
    out = []
    for name in NAMED:
        for k in range(n_seeds):
            out.append((name, make_named(name, C.TEST_SEED_BASE + k)))
    for k in range(n_seeds):
        out.append(("random_heldout", make_random(C.TEST_SEED_BASE + 1000 + k, tag="test")))
    return out


def val_set(n=6):
    return [make_random(C.VAL_SEED_BASE + k, tag="val") for k in range(n)]
