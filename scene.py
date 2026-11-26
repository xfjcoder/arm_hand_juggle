"""UR10e arm + Shadow Hand (E3M5, right) + juggling balls, built with MjSpec from mujoco_menagerie.

The hand's forearm is bolted to the UR10e tool flange (attachment_site), forearm axis along the
flange's z, so the fingers point out of the flange. In the forearm frame the palm faces +x.

The arm's six actuators are turned into torque motors (ctrl = joint torque, clipped to the real
UR10e limits 330/330/150/56/56/56 N m); the controller does computed-torque tracking itself. Real
UR10e joint SPEED limits are not in the model, so the planner must respect them (JOINT_VMAX).
"""
from pathlib import Path

import mujoco as mj
import numpy as np

MENAGERIE = Path.home() / "robo" / "mujoco_menagerie"
ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
              "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ARM_ACTS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")
TORQUE_MAX = np.array([330, 330, 150, 56, 56, 56], float)
# UR10e datasheet: base and shoulder 120 deg/s, elbow and wrists 180 deg/s.
JOINT_VMAX = np.radians([120, 120, 180, 180, 180, 180])

BALL_RADIUS = 0.03          # 6 cm juggling ball
BALL_MASS = 0.10
TIMESTEP = 0.002


def build_spec(n_balls=1):
    arm = mj.MjSpec.from_file(str(MENAGERIE / "universal_robots_ur10e" / "ur10e.xml"))
    hand = mj.MjSpec.from_file(str(MENAGERIE / "shadow_hand" / "right_hand.xml"))
    taken = {mt.name for mt in arm.materials}         # both models define e.g. "black": rename the hand's
    for mt in hand.materials:
        if mt.name in taken:
            for g in hand.geoms:
                if g.material == mt.name:
                    g.material = "rh_" + mt.name
            mt.name = "rh_" + mt.name
    taken = {ms.name or Path(ms.file).stem for ms in arm.meshes}
    hand_meshdir = MENAGERIE / "shadow_hand" / hand.meshdir
    for ms in hand.meshes:
        ms.name = ms.name or Path(ms.file).stem       # unnamed meshes are named after their file
        ms.file = str(hand_meshdir / ms.file)         # absolute, so the arm's meshdir does not apply
        if ms.name in taken:
            for g in hand.geoms:
                if g.meshname == ms.name:
                    g.meshname = "rh_" + ms.name
            ms.name = "rh_" + ms.name
    hand_root = hand.body("rh_forearm")
    hand_root.quat = [1, 0, 0, 0]                     # forearm z = flange z
    arm.body("base").quat = [1, 0, 0, 0]              # menagerie faces -x; face the juggling zone at +x
    site = arm.site("attachment_site")
    arm.attach(hand, site=site, prefix="")

    arm.option.timestep = TIMESTEP
    arm.option.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    arm.option.cone = mj.mjtCone.mjCONE_ELLIPTIC
    arm.option.impratio = 10
    arm.visual.global_.offwidth = 1280
    arm.visual.global_.offheight = 720

    for name, tmax in zip(ARM_ACTS, TORQUE_MAX):
        a = arm.actuator(name)
        a.gaintype = mj.mjtGain.mjGAIN_FIXED
        a.gainprm[:3] = [1, 0, 0]
        a.biastype = mj.mjtBias.mjBIAS_NONE
        a.biasprm[:3] = [0, 0, 0]
        a.ctrlrange = [-tmax, tmax]
        a.forcerange = [-tmax, tmax]

    wb = arm.worldbody
    tex = arm.add_texture(name="grid", type=mj.mjtTexture.mjTEXTURE_2D, builtin=mj.mjtBuiltin.mjBUILTIN_CHECKER,
                          rgb1=[.25, .3, .35], rgb2=[.2, .24, .28], width=512, height=512)
    arm.add_material(name="grid", textures=["", "grid"], texrepeat=[6, 6], reflectance=0.1)
    wb.add_geom(name="floor", type=mj.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.05], material="grid", pos=[0, 0, -0.8])
    wb.add_geom(name="pedestal", type=mj.mjtGeom.mjGEOM_CYLINDER, size=[0.1, 0.4, 0], pos=[0, 0, -0.4],
                rgba=[.35, .35, .38, 1], contype=0, conaffinity=0)   # visual only
    wb.add_light(pos=[0.5, -1.0, 2.5], dir=[0, 0.3, -1], diffuse=[.8, .8, .8])
    wb.add_light(pos=[1.5, 1.0, 2.0], dir=[-0.5, -0.3, -1], diffuse=[.4, .4, .4])
    colours = ([.9, .2, .2, 1], [.2, .6, .95, 1], [.95, .8, .1, 1], [.3, .85, .3, 1])
    for i in range(n_balls):
        b = wb.add_body(name=f"ball{i}", pos=[0.7, 0.3 * i, 0.2 + 0.1 * i])
        b.add_freejoint(name=f"ball{i}")
        b.add_geom(name=f"ball{i}", type=mj.mjtGeom.mjGEOM_SPHERE, size=[BALL_RADIUS, 0, 0], mass=BALL_MASS,
                   rgba=colours[i % 4], condim=4, friction=[1.0, 0.01, 0.001], priority=1)
    cam = wb.add_camera(name="side", pos=[1.0, -2.0, 0.9], xyaxes=[0.9, 0.44, 0, -0.1, 0.2, 0.97])
    return arm


def build(n_balls=1):
    return build_spec(n_balls).compile()


class Ids:
    """Indices into qpos / qvel / ctrl for the arm, hand and balls."""

    def __init__(self, m):
        self.arm_q = np.array([m.jnt_qposadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_v = np.array([m.jnt_dofadr[m.joint(j).id] for j in ARM_JOINTS])
        self.arm_u = np.array([m.actuator(a).id for a in ARM_ACTS])
        self.hand_u = np.array([i for i in range(m.nu) if i not in self.arm_u])
        self.palm = m.body("rh_palm").id
        self.forearm = m.body("rh_forearm").id
        self.balls = [m.body(f"ball{i}").id for i in range(sum(1 for i in range(m.nbody)
                                                              if m.body(i).name.startswith("ball")))]
        self.ball_q = [m.jnt_qposadr[m.joint(f"ball{i}").id] for i in range(len(self.balls))]
        self.ball_v = [m.jnt_dofadr[m.joint(f"ball{i}").id] for i in range(len(self.balls))]


if __name__ == "__main__":
    m = build(1)
    d = mj.MjData(m)
    ids = Ids(m)
    mj.mj_forward(m, d)
    print(f"nq {m.nq} nv {m.nv} nu {m.nu} bodies {m.nbody}  arm+hand mass "
          f"{m.body_subtreemass[m.body('base').id]:.2f} kg, hand {m.body_subtreemass[ids.forearm]:.2f} kg")
    R = d.xmat[ids.forearm].reshape(3, 3)
    print("forearm z (finger direction) in world:", R[:, 2].round(3), " palm normal (forearm x):", R[:, 0].round(3))
