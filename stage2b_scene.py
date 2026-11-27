"""Stage 2b: the two-arm robot (scene.build(arms=2)) in the Stage 1 cocked pose (top row) and release pose
(bottom row), front and three-quarter views, plus a top view. The left arm is the exact mirror of the right
(kin.mirror).

    PYTHONPATH=. MUJOCO_GL=egl python3 stage2b_scene.py
"""
import mujoco as mj, numpy as np, scene, juggle, human, kin
from PIL import Image
m = scene.build(2, human=True, arms=2)
d = mj.MjData(m)
R = juggle.Juggler(side="R", m=m, d=d); L = juggle.Juggler(side="L", m=m, d=d)
# Stage 1 posture: cocked (the lowest pose) and release
import stage1_juggle, cem_throw
x, cp, off = stage1_juggle.load("runs/catch5_v443.json")
th, br, q_rel = cem_throw.segments(x, 0.35, 0.2)
imgs = []
r = mj.Renderer(m, 600, 800)
for q in (th(0)[0], q_rel):
    d.qpos[:] = m.qpos0
    for J, qq in ((R, q), (L, kin.mirror(q))):
        d.qpos[J.ids.arm_q] = qq
        for u in J.ids.hand_u:
            j = m.actuator_trnid[u, 0]
            if m.actuator_trntype[u] == mj.mjtTrn.mjTRN_JOINT:
                d.qpos[m.jnt_qposadr[j]] = J.hold[u]
    mj.mj_forward(m, d)
    for i, J in enumerate((R, L)):
        p, Rp = kin.palm_pose(m, d, J.ids)
        bq = J.ids.ball_q[i]
        d.qpos[bq:bq + 3] = p + Rp @ np.array([0, -0.06, 0.01])
    mj.mj_forward(m, d)
    pr, pl_ = kin.palm_pose(m, d, R.ids)[0], kin.palm_pose(m, d, L.ids)[0]
    arm_contacts = [(m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name) for c in d.contact[:d.ncon]
                    if not m.body(m.geom_bodyid[c.geom1]).name.startswith("ball") and not m.body(m.geom_bodyid[c.geom2]).name.startswith("ball")]
    print("palms R", pr.round(3), "L", pl_.round(3), "contacts", arm_contacts[:6])
    row = []
    for az, el, dist in ((180, -10, 3.4), (130, -15, 3.4)):
        cam = mj.MjvCamera(); cam.lookat[:] = [0.6, 0, -0.1]; cam.distance, cam.azimuth, cam.elevation = dist, az, el
        r.update_scene(d, cam); row.append(r.render())
    imgs.append(np.concatenate(row, 1))
cam = mj.MjvCamera(); cam.lookat[:] = [0.5, 0, -0.2]; cam.distance, cam.azimuth, cam.elevation = 3.0, 180, -89
r.update_scene(d, cam); top = r.render()
img = np.concatenate(imgs, 0)
Image.fromarray(img).save("media/two_arm_scene.png")
Image.fromarray(top).save("media/two_arm_top.png")
print(img.shape)
