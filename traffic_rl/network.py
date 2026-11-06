"""Build the 4-way intersection with netconvert and derive the signal phases.

Each approach has 3 incoming lanes:
  lane 0: right + straight,  lane 1: straight,  lane 2: protected left.
Phases (all protected, no permissive conflicts):
  0 NS_through  1 NS_left  2 EW_through  3 EW_left
"""
import os
import subprocess

import sumolib

from . import config as C

# (origin, turn) -> destination
DEST = {
    ("N", "r"): "W", ("N", "s"): "S", ("N", "l"): "E",
    ("S", "r"): "E", ("S", "s"): "N", ("S", "l"): "W",
    ("E", "r"): "N", ("E", "s"): "W", ("E", "l"): "S",
    ("W", "r"): "S", ("W", "s"): "E", ("W", "l"): "N",
}
TURN_OF = {(o, d): t for (o, t), d in DEST.items()}
POS = {"N": (0, C.EDGE_LENGTH), "S": (0, -C.EDGE_LENGTH),
       "E": (C.EDGE_LENGTH, 0), "W": (-C.EDGE_LENGTH, 0)}

# incoming-lane indices served by each phase
#   lanes are ordered N0 N1 N2 E0 E1 E2 S0 S1 S2 W0 W1 W2
PHASE_LANES = [
    [0, 1, 6, 7],    # NS through/right
    [2, 8],          # NS left
    [3, 4, 9, 10],   # EW through/right
    [5, 11],         # EW left
]

PHASE_MOVES = [
    ({"N", "S"}, {"s", "r"}),
    ({"N", "S"}, {"l"}),
    ({"E", "W"}, {"s", "r"}),
    ({"E", "W"}, {"l"}),
]


def _write(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def build_network():
    os.makedirs(C.NET_DIR, exist_ok=True)
    nodes = ['<nodes>', f'  <node id="{C.TLS_ID}" x="0" y="0" type="traffic_light"/>']
    for a, (x, y) in POS.items():
        nodes.append(f'  <node id="{a}" x="{x}" y="{y}" type="priority"/>')
    nodes.append('</nodes>')

    edges = ['<edges>']
    for a in C.APPROACHES:
        edges.append(f'  <edge id="{a}2C" from="{a}" to="C" numLanes="3" speed="{C.SPEED}"/>')
        edges.append(f'  <edge id="C2{a}" from="C" to="{a}" numLanes="3" speed="{C.SPEED}"/>')
    edges.append('</edges>')

    cons = ['<connections>']
    for o in C.APPROACHES:
        r, s, l = DEST[(o, "r")], DEST[(o, "s")], DEST[(o, "l")]
        cons += [
            f'  <connection from="{o}2C" to="C2{r}" fromLane="0" toLane="0"/>',
            f'  <connection from="{o}2C" to="C2{s}" fromLane="0" toLane="0"/>',
            f'  <connection from="{o}2C" to="C2{s}" fromLane="1" toLane="1"/>',
            f'  <connection from="{o}2C" to="C2{l}" fromLane="2" toLane="2"/>',
        ]
    cons.append('</connections>')

    nod, edg, con = (os.path.join(C.NET_DIR, f) for f in ("n.nod.xml", "e.edg.xml", "c.con.xml"))
    _write(nod, "\n".join(nodes))
    _write(edg, "\n".join(edges))
    _write(con, "\n".join(cons))
    subprocess.run([C.NETCONVERT_BIN, "-n", nod, "-e", edg, "-x", con, "-o", C.NET_FILE,
                    "--no-turnarounds", "true", "--tls.default-type", "static",
                    "--no-warnings", "true"], check=True)
    write_tll()
    return C.NET_FILE


def phase_strings(net_file=None):
    """Return (green_states, yellow_states, incoming_lane_ids)."""
    net = sumolib.net.readNet(net_file or C.NET_FILE)
    conns = net.getTLS(C.TLS_ID).getConnections()
    n_links = max(c[2] for c in conns) + 1
    greens, yellows = [], []
    for origins, turns in PHASE_MOVES:
        g = ["r"] * n_links
        for in_lane, out_lane, idx in conns:
            o = in_lane.getEdge().getID()[0]
            d = out_lane.getEdge().getID()[2]
            if o in origins and TURN_OF[(o, d)] in turns:
                g[idx] = "G"
        greens.append("".join(g))
        yellows.append("".join("y" if ch == "G" else "r" for ch in g))
    lanes = [f"{a}2C_{i}" for a in C.APPROACHES for i in range(3)]
    return greens, yellows, lanes


def write_tll():
    """Static (naive 30s) and actuated programs, used by SUMO's native controllers."""
    greens, yellows, _ = phase_strings()
    out = ['<additional>']
    out.append(f'  <tlLogic id="{C.TLS_ID}" type="actuated" programID="actuated" offset="0">')
    out.append('    <param key="max-gap" value="3.0"/>')
    out.append('    <param key="detector-gap" value="2.0"/>')
    out.append('    <param key="passing-time" value="2.0"/>')
    for i, (g, y) in enumerate(zip(greens, yellows)):
        mx = 60 if i % 2 == 0 else 30
        out.append(f'    <phase duration="{C.DELTA}" minDur="{C.DELTA}" maxDur="{mx}" state="{g}"/>')
        out.append(f'    <phase duration="{C.YELLOW}" state="{y}"/>')
    out.append('  </tlLogic>')
    out.append('</additional>')
    _write(C.TLL_FILE, "\n".join(out))


if __name__ == "__main__":
    build_network()
    g, y, lanes = phase_strings()
    for name, s in zip(C.PHASE_NAMES, g):
        print(f"{name:12s} {s}")
    print("incoming lanes:", lanes)
