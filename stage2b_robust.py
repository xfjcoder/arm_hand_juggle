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


def run(seed, throws, kick, sensors, noise, balls=2):
    if not _W:
        JR, JL = stage2b_juggle.make(balls)
        _W.update(JR=JR, JL=JL, cy=stage2b_juggle.load_cycle(JR))
    r = stage2b_juggle.juggle2b(_W["JR"], _W["JL"], _W["cy"], n_balls=balls, throws=throws, seed=seed, kick=kick,
                                preset=sensors, noise=noise)
    rel = r.pop("releases")
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
    a = ap.parse_args()
    R = evalpool.evaluate(run, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()),
                          kwargs=dict(throws=a.throws, kick=a.kick, sensors=a.sensors, noise=a.noise, balls=a.balls))
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
