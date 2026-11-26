"""Stage 2, one hand: what happened, side by side, in slow motion (one catch -> carry -> throw each).

    PYTHONPATH=.:../robolab python3 stage2_onehand_gif.py media_full/stage2_onehand.gif
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
    ("1. gate cycle, open palm", np.r_[B["z0"], 0.05, 0.03, 0.0, 0.0]),
    ("2. cage grip, CEM v1", np.array(json.load(open("runs/cycle2_v1.json"))["x"])),
    ("3. cage grip, CEM v2 (limits enforced)", np.array(json.load(open("runs/cycle2_v2.json"))["x"])),
]
inc = dict(kick=np.zeros(3), off=np.zeros(3), spin=np.zeros(3))
J.renderer = mj.Renderer(J.m, 480, 400)
render = J.render
J.render = lambda: render(distance=2.3, azimuth=160, elevation=-8, lookat=[1.2, 0.27, -0.05])
font = ImageFont.load_default(size=17)
panels = []
for name, x in VARIANTS:
    cy = sc.build(J, B["pl"], B["base"], x)
    frames = []
    r = sc.unit(J, B["pl"], cy, inc, frames=frames, every=4)
    for _ in range(40):                                 # let the thrown ball fly a little after the unit ends
        J.control_joint(J.d.qpos[J.ids.arm_q].copy(), np.zeros(6), np.zeros(6))
        mj.mj_step(J.m, J.d)
        if _ % 4 == 0:
            frames.append(J.render())
    if r["held"]:
        v = np.round(r["v"], 1).tolist()
        res = f"thrown at {v} m/s (want [0, 0.2, 4.5])"
    else:
        res = "ball lost in the hand"
    lim = f"peak joint speed {max(r['qd_ratio'], r['peak']):.2f} x limit"
    out = []
    for f in frames:
        im = Image.fromarray(f)
        dr = ImageDraw.Draw(im)
        dr.text((8, 6), name, fill=(255, 255, 255), font=font)
        dr.text((8, 28), res, fill=(255, 220, 120), font=font)
        dr.text((8, 50), lim, fill=(255, 120, 120) if max(r['qd_ratio'], r['peak']) > 0.97 else (140, 255, 140), font=font)
        out.append(np.asarray(im))
    panels.append(out)
    print(name, r["held"], lim, res, flush=True)
J.renderer.close()
n = max(len(p) for p in panels)
panels = [p + [p[-1]] * (n - len(p)) for p in panels]
frames = [np.concatenate([p[i] for p in panels], axis=1) for i in range(n)]
frames += [frames[-1]] * 20
from robolab.media import save_gif
save_gif(frames, sys.argv[1], fps=15)                   # 8 ms per frame at 15 fps: about 1/8 speed
