"""Stage 2b robustness: the two-arm two-ball cascade over many seeds (velocity kicks at every release,
sensing preset ideal / real). Runs capped at --throws catches.

    PYTHONPATH=.:../robolab python3 stage2b_robust.py --n 120 --sensors ideal --out logs/stage2b_ideal.jsonl
"""
import argparse
import json
import os

import numpy as np

import stage2b_juggle
from robolab import evalpool

_W = {}


def run(seed, throws, kick, sensors, noise, balls=2, ghost=False, cycle=None, start=None, aim=None):
    if not _W:
        JR, JL = stage2b_juggle.make(balls)
        _W.update(JR=JR, JL=JL, cy=stage2b_juggle.load_cycle(JR, *((cycle, start) if cycle else ())))
        if ghost:                                    # balls pass through each other (still collide with the rest)
            for i in JR.ids.balls:
                JR.m.geom_contype[JR.m.body_geomadr[i]] = 2
    r = stage2b_juggle.juggle2b(_W["JR"], _W["JL"], _W["cy"], n_balls=balls, throws=throws, seed=seed, kick=kick,
                                preset=sensors, noise=noise, aim=aim)
    rel = r.pop("releases")
    A = r.pop("aims")
    r["aim"] = dict(n=len(A), moved=sum(1 for x in A if x["dp"]), dp_max=max([x["dp"] for x in A] or [0.0]),
                    gap0_min=min([x["gap0"] for x in A] or [1.0]), gap_min=min([x["gap"] for x in A] or [1.0]),
                    pk0_max=max([x.get("pk0", 0.0) for x in A] or [0.0]), pk_max=max([x.get("pk", 0.0) for x in A] or [0.0]))
    r["land_err"] = [x["land_err"] + [x["t_err"]] for x in rel]
    r.update(seed=seed, all=int(r["catches"] >= throws), within_limits=int(r["qd_ratio"] <= 1.0 and r["tau_sat"] <= 0.01))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--balls", type=int, default=2)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--throws", type=int, default=30)
    ap.add_argument("--kick", type=float, default=0.05)
    ap.add_argument("--sensors", default="ideal", choices=["ideal", "real"])
    ap.add_argument("--noise", type=float, default=0.002)
    ap.add_argument("--out", default="logs/stage2b_ideal.jsonl")
    ap.add_argument("--ghost", action="store_true", help="no ball-ball collisions (diagnostic)")
    ap.add_argument("--run", help="cycle run (default: stage2b_juggle.load_cycle's)")
    ap.add_argument("--start", help="start-up throw run for --run")
    ap.add_argument("--aim-start", type=float, default=None,
                    help="offset limit on the start-up throws (m); 0: none; default: as DMAX")
    ap.add_argument("--aim-reach", type=float, default=None, metavar="W",
                    help="reach-aware offset (stage2b_juggle.aim_reach), W: cost per dmax of offset")
    ap.add_argument("--aim", type=float, nargs=3, metavar=("MARGIN", "DMAX", "LEAD"),
                    help="offset each release point to clear the incoming ball (stage2b_juggle.aim_offset)")
    a = ap.parse_args()
    R = evalpool.evaluate(run, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()),
                          kwargs=dict(throws=a.throws, kick=a.kick, sensors=a.sensors, noise=a.noise, balls=a.balls,
                                      ghost=a.ghost, cycle=a.run, start=a.start,
                                      aim=dict(margin=a.aim[0], dmax=a.aim[1], lead=a.aim[2],
                                               start=a.aim_start != 0, start_dmax=a.aim_start or a.aim[1], reach=a.aim_reach)
                                      if a.aim else None))
    with open(a.out, "w") as f:
        for r in R:
            f.write(json.dumps(r) + "\n")
    c = np.array([r["catches"] for r in R])
    print(f"{a.balls} balls, {a.sensors}, seeds {a.seed0}+, n {a.n}: reach {a.throws}: {sum(r['all'] for r in R)}/{a.n}  "
          f"catches median {np.median(c):.0f} mean {c.mean():.1f} quartiles {np.percentile(c, [25, 75]).tolist()}  "
          f"within limits {sum(r['within_limits'] for r in R)}/{a.n}  max qd {max(r['qd_ratio'] for r in R):.3f}")
    why = {}
    for r in R:
        if r["why"]:
            why[r["why"]] = why.get(r["why"], 0) + 1
    print("why runs ended early:", why)


if __name__ == "__main__":
    main()
