"""SUMO environment for a single signalised intersection.

Every controller (RL, fixed-time, max-pressure) drives the signal through the
same `step(phase, green)` call, so yellow handling and minimum green are
identical.  SUMO's native actuated controller runs via `run_native()`.

Reward = -(halting vehicles on incoming lanes + vehicles waiting to enter),
summed over every simulated second of the step.  Summed over an episode this is
exactly minus the total waiting time, i.e. the metric we evaluate on.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C
from .network import PHASE_LANES, phase_strings

REWARD_SCALE = 1 / 100.0
LANE_CAP = C.EDGE_LENGTH / 7.5  # vehicles that fit on a lane


class SumoEnv:
    n_actions = 4

    def __init__(self, gui=False, sumo_seed=0, max_time=C.MAX_TIME, gui_args=(), hook=None):
        self.gui = gui
        self.gui_args = list(gui_args)  # extra sumo-gui options (window position, delay, ...)
        self.hook = hook                # called as hook(env) after every simulated second
        self.max_time = max_time  # training uses a shorter cap; evaluation always C.MAX_TIME
        if gui:
            import traci
        else:
            import libsumo as traci
        self.traci = traci
        self.sumo_seed = sumo_seed
        self.greens, self.yellows, self.lanes = phase_strings()
        self.obs_dim = 3 * len(self.lanes) + self.n_actions + 1
        self.running = False
        self.tripinfo = None

    # ------------------------------------------------------------------ setup
    def _start(self, route_file):
        self.close()
        fd, self.tripinfo = tempfile.mkstemp(suffix=".tripinfo.xml")
        os.close(fd)
        cmd = [C.SUMO_GUI_BIN if self.gui else C.SUMO_BIN,
               "-n", C.NET_FILE, "-r", route_file, "-a", C.TLL_FILE,
               "--seed", str(self.sumo_seed),
               "--time-to-teleport", "-1",
               "--waiting-time-memory", "100000",
               "--tripinfo-output", self.tripinfo,
               "--tripinfo-output.write-unfinished", "true",
               "--tripinfo-output.write-undeparted", "true",
               "--end", str(C.MAX_TIME),
               "--no-step-log", "true", "--no-warnings", "true"]
        if self.gui:
            cmd += ["--start", "--quit-on-end", "-g", C.VIEW_FILE] + self.gui_args
        self.traci.start(cmd)
        self.running = True
        self.t = 0
        self.total_cost = 0.0
        self.overrides = 0

    def reset(self, route_file):
        self._start(route_file)
        self.phase = 0
        self.time_in_phase = 0
        self.traci.trafficlight.setRedYellowGreenState(C.TLS_ID, self.greens[0])
        return self._obs()

    # ------------------------------------------------------------- dynamics
    def _sim_second(self):
        tr = self.traci
        tr.simulationStep()
        self.t += 1
        cost = sum(tr.lane.getLastStepHaltingNumber(l) for l in self.lanes)
        cost += len(tr.simulation.getPendingVehicles())
        self.total_cost += cost
        if self.hook:
            self.hook(self)
        return cost

    def drained(self):
        return self.t >= C.DEMAND_END and self.traci.simulation.getMinExpectedNumber() == 0

    def done(self):
        return self.t >= self.max_time or self.drained()

    def lane_head_wait(self):
        """Longest current stop (s) of any vehicle on each incoming lane."""
        v = self.traci.vehicle
        return np.array([max((v.getWaitingTime(x) for x in self.traci.lane.getLastStepVehicleIDs(l)),
                             default=0.0) for l in self.lanes])

    def _guard(self, action):
        """Max-red safety rule: a phase whose waiting vehicle has been stopped
        >= MAX_RED seconds is served next, whatever the controller asked for."""
        hw = self.lane_head_wait()
        phase_wait = [hw[idx].max() for idx in PHASE_LANES]
        worst = int(np.argmax(phase_wait))
        if worst != action and worst != self.phase and phase_wait[worst] >= C.MAX_RED:
            self.overrides += 1
            return worst
        return action

    def step(self, action, green=C.DELTA):
        tl = self.traci.trafficlight
        cost = 0.0
        action = self._guard(action)
        self.last_action = action
        if action != self.phase:
            tl.setRedYellowGreenState(C.TLS_ID, self.yellows[self.phase])
            for _ in range(C.YELLOW):
                cost += self._sim_second()
            self.phase, self.time_in_phase = action, 0
            tl.setRedYellowGreenState(C.TLS_ID, self.greens[action])
        for _ in range(green):
            cost += self._sim_second()
            if self.done():
                break
        self.time_in_phase += green
        return self._obs(), -cost * REWARD_SCALE, self.done()

    def run_native(self, route_file, program="actuated"):
        self._start(route_file)
        self.traci.trafficlight.setProgram(C.TLS_ID, program)
        while not self.done():
            self._sim_second()

    # ----------------------------------------------------------- observation
    def lane_halting(self):
        return np.array([self.traci.lane.getLastStepHaltingNumber(l) for l in self.lanes], float)

    def lane_counts(self):
        return np.array([self.traci.lane.getLastStepVehicleNumber(l) for l in self.lanes], float)

    def _obs(self):
        onehot = np.zeros(self.n_actions)
        onehot[self.phase] = 1
        return np.concatenate([
            self.lane_counts() / LANE_CAP,
            self.lane_halting() / LANE_CAP,
            np.minimum(self.lane_head_wait(), C.MAX_RED) / C.MAX_RED,
            onehot,
            [min(self.time_in_phase, 60) / 60.0],
        ]).astype(np.float32)

    # --------------------------------------------------------------- metrics
    def close(self):
        if self.running:
            self.traci.close()
            self.running = False

    def metrics(self):
        """Close the simulation and summarise tripinfo (incl. unfinished vehicles)."""
        self.close()
        waits, delays, losses, unfinished = [], [], [], 0
        for ti in ET.parse(self.tripinfo).getroot().iter("tripinfo"):
            dd = float(ti.get("departDelay", 0))
            wt = float(ti.get("waitingTime", 0))
            if float(ti.get("arrival", -1)) < 0:
                unfinished += 1
            waits.append(wt + dd)          # time stopped at the signal + time queued to enter
            losses.append(float(ti.get("timeLoss", 0)) + dd)
            delays.append(dd)
        os.remove(self.tripinfo)
        return {
            "avg_wait": float(np.mean(waits)) if waits else 0.0,
            "avg_time_loss": float(np.mean(losses)) if losses else 0.0,
            "avg_depart_delay": float(np.mean(delays)) if delays else 0.0,
            "p95_wait": float(np.percentile(waits, 95)) if waits else 0.0,
            "max_wait": float(np.max(waits)) if waits else 0.0,
            "vehicles": len(waits),
            "unfinished": unfinished,
            "sim_time": self.t,
            "guard_overrides": self.overrides,
        }
