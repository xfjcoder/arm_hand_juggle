"""Stage 2, one hand: close-up of one catch -> carry -> throw for each cycle tried, side by side, about
1/8 speed (stage2_cem.unit: an incoming ball on the nominal path, no disturbances).

    PYTHONPATH=.:../robolab python3 stage2_closeup_gif.py media_full/stage2_closeup.gif
"""
import json
import sys

import mujoco as mj
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import stage2_cem as sc

B = sc._base()
J = B["J"]
VARIANTS = [
    ("1. open palm", np.r_[B["z0"], 0.05, 0.03, 0.0, 0.0]),
    ("2. cage grip, CEM v1", np.array(json.load(open("runs/cycle2_v1.json"))["x"])),
    ("3. cage grip, CEM v2", np.array(json.load(open("runs/cycle2_v2.json"))["x"])),
]
inc = dict(kick=np.zeros(3), off=np.zeros(3), spin=np.zeros(3))
J.renderer = mj.Renderer(J.m, 420, 420)
render = J.render
J.render = lambda: render(distance=1.1, azimuth=150, elevation=-12, lookat=[1.25, 0.28, -0.2])
font = ImageFont.load_default(size=17)
panels = []
for name, x in VARIANTS:
    cy = sc.build(J, B["pl"], B["base"], x)
    frames = []
    r = sc.unit(J, B["pl"], cy, inc, frames=frames, every=4)
    for k in range(60):
        J.control_joint(J.d.qpos[J.ids.arm_q].copy(), np.zeros(6), np.zeros(6))
        mj.mj_step(J.m, J.d)
        if k % 4 == 0:
            frames.append(J.render())
    res = f"thrown up at {r['v'][2]:.1f} m/s (want 4.5)" if r["held"] else "ball lost in the carry"
    pk = max(r["qd_ratio"], r["peak"])
    lines = [(name, (255, 255, 255)), (res, (255, 220, 120)),
             (f"peak joint speed {pk:.2f} x limit", (255, 120, 120) if pk > 0.97 else (140, 255, 140))]
    out = []
    for f in frames:
        im = Image.fromarray(f)
        dr = ImageDraw.Draw(im)
        for i, (ln, col) in enumerate(lines):
            dr.text((8, 6 + 22 * i), ln, fill=col, font=font)
        out.append(np.asarray(im))
    panels.append(out)
    print(name, res, pk, flush=True)
J.renderer.close()
n = max(len(p) for p in panels)
panels = [p + [p[-1]] * (n - len(p)) for p in panels]
frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)] + []
from robolab.media import save_gif
save_gif(frames, sys.argv[1], fps=15)
