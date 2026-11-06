"""Shared paths and constants."""
import os

import sumo  # pip package "eclipse-sumo"

os.environ.setdefault("SUMO_HOME", sumo.SUMO_HOME)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NET_DIR = os.path.join(ROOT, "network")
NET_FILE = os.path.join(NET_DIR, "intersection.net.xml")
TLL_FILE = os.path.join(NET_DIR, "programs.tll.xml")
VIEW_FILE = os.path.join(NET_DIR, "demo.view.xml")
ROUTE_DIR = os.path.join(ROOT, "routes")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")

SUMO_BIN = os.path.join(sumo.SUMO_HOME, "bin", "sumo")
SUMO_GUI_BIN = os.path.join(sumo.SUMO_HOME, "bin", "sumo-gui")
NETCONVERT_BIN = os.path.join(sumo.SUMO_HOME, "bin", "netconvert")

TLS_ID = "C"
APPROACHES = ["N", "E", "S", "W"]
EDGE_LENGTH = 300.0
SPEED = 13.89  # m/s (50 km/h)

# Signal timing (identical for every controller)
YELLOW = 3          # s
DELTA = 5           # s, RL / max-pressure decision interval (= min green)
MAX_RED = 120      # s, safety guard: a vehicle stopped this long forces its phase
DEMAND_END = 3600  # s, vehicles are inserted during the first hour
MAX_TIME = 7200     # s, hard cap; episodes normally end when the network drains

PHASE_NAMES = ["NS_through", "NS_left", "EW_through", "EW_left"]

# Test/validation/train seed ranges never overlap
TRAIN_SEED_BASE = 0
VAL_SEED_BASE = 100_000
TEST_SEED_BASE = 200_000
