"""Stage 2, one hand: what happened, side by side, in slow motion: the two-ball loop (start-up throw, the
second ball fed one cycle behind, then catch / carry / throw) with each cycle tried, until it fails.

    PYTHONPATH=.:../robolab python3 stage2_onehand_gif.py media_full/stage2_onehand.gif
"""
import json
import sys

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import juggle
import stage2_juggle
from planner import Planner

VARIANTS = [
    ("1. gate cycle, open palm", None),
    ("2. cage grip, CEM v1", json.load(open("runs/cycle2_v1.json"))["x"]),
    ("3. cage grip, CEM v2 (limits enforced)", json.load(open("runs/cycle2_v2.json"))["x"]),
]
font = ImageFont.load_default(size=17)
panels = []
for name, x in VARIANTS:
    J = juggle.Juggler(2, human=True)
    pl = Planner(J)
    cy = stage2_juggle.load_cycle(J, pl, x=x)
    J.m.geom_rgba[J.m.body_geomadr[J.ids.balls[1]]] = [0.2, 0.6, 1.0, 1.0]   # ball 2 bright blue
    J.renderer = mj.Renderer(J.m, 480, 400)
    render = J.render
    J.render = lambda: render(distance=2.7, azimuth=160, elevation=-6, lookat=[1.2, 0.27, 0.2])
    frames = []
    r = stage2_juggle.juggle2(J, cy, cycles=10, seed=1, preset="ideal", noise=0.0, frames=frames, every=4)
    for k in range(150):                                # keep going a little: watch the balls fall
        J.control_joint(J.d.qpos[J.ids.arm_q].copy(), np.zeros(6), np.zeros(6))
        mj.mj_step(J.m, J.d)
        if k % 4 == 0:
            frames.append(J.render())
    J.renderer.close()
    rel = ", ".join(f"{v[2]:.1f}" for v in r["releases"][:3]) or "(left the hand before the release window; not logged)"
    lines = [name, f"throw up-speeds: {rel}" + (" m/s" if r["releases"] else "") + " (want 4.5)", f"catches {r['catches']}, ended: {r['why']}",
             f"peak joint speed {r['qd_ratio']:.2f} x limit"]
    out = []
    for f in frames:
        im = Image.fromarray(f)
        dr = ImageDraw.Draw(im)
        for i, (ln, col) in enumerate(zip(lines, [(255, 255, 255), (255, 220, 120), (255, 220, 120),
                                                   (255, 120, 120) if r["qd_ratio"] > 0.97 else (140, 255, 140)])):
            dr.text((8, 6 + 22 * i), ln, fill=col, font=font)
        out.append(np.asarray(im))
    panels.append(out)
    print(lines, flush=True)
n = max(len(p) for p in panels)
panels = [p + [p[-1]] * (n - len(p)) for p in panels]
frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
from robolab.media import save_gif
save_gif(frames, sys.argv[1], fps=15)                   # 8 ms per frame at 15 fps: about 1/8 speed
