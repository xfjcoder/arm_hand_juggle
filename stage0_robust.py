"""Stage 0 check: is the single toss and catch robust, not a lucky run?

Per seed: ball dropped up to 1.5 cm off-centre, throw speed scaled by U(0.9, 1.1), 2 mm noise on the
measured ball position at every re-plan. Pose A (1.0, 0, 0.3), stroke 0.3 m, nominal 2.0 m/s.

    PYTHONPATH=. python3 stage0_robust.py --n 30
"""
import argparse
import os

import numpy as np

import juggle
import stage0_toss
from robolab import evalpool, stats


def episode(seed, v=2.0, stroke=0.3, t_throw=0.36, noise=0.002, **kw):
    rng = np.random.default_rng(seed)
    J = juggle.Juggler(1)
    off = rng.uniform(-0.015, 0.015, 2)
    J.reset(ball_offset=(off[0], off[1], 0.06))
    L = stage0_toss.toss(J, v_r=v * rng.uniform(0.9, 1.1), stroke=stroke, t_throw=t_throw, noise=noise, rng=rng, **kw)
    rel = L["release"]["ball_v"] if L["release"] else [0, 0, 0]
    return dict(held=int(L["held"]), caught=int(L["caught"]), release_vz=rel[2], apex=L["apex_above_release"],
                qd_ratio=L["qd_ratio"], within_limits=int(L["qd_ratio"] <= 1.0), tau_sat=L["tau_sat_frac"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=50000)
    a = ap.parse_args()
    R = evalpool.evaluate(episode, range(a.seed0, a.seed0 + a.n), procs=min(12, os.cpu_count()))
    evalpool.save_jsonl(R, "logs/stage0_robust.jsonl")
    print(evalpool.summarise(R, rates=["held", "caught", "within_limits"], means=["release_vz", "apex", "qd_ratio", "tau_sat"]))
