"""Baseline controllers and a common episode runner."""
import numpy as np

from .network import PHASE_LANES


def run_fixed(env, route_file, greens=(30, 15, 30, 15)):
    """Fixed-time cycle; greens = seconds of green per phase (yellow added on top)."""
    env.reset(route_file)
    done, k = False, 0
    while not done:
        _, _, done = env.step(k % 4, green=greens[k % 4])
        k += 1
    return env.metrics()


def max_pressure_action(env):
    """Serve the phase with the most queued vehicles (isolated junction => max-pressure)."""
    q = env.lane_halting() + 0.5 * env.lane_counts()
    pressure = [q[idx].sum() for idx in PHASE_LANES]
    return int(np.argmax(pressure))


def run_policy(env, route_file, policy):
    obs = env.reset(route_file)
    done = False
    while not done:
        obs, _, done = env.step(policy(obs, env))
    return env.metrics()


def run_max_pressure(env, route_file):
    return run_policy(env, route_file, lambda obs, e: max_pressure_action(e))


def run_actuated(env, route_file):
    env.run_native(route_file, "actuated")
    return env.metrics()
