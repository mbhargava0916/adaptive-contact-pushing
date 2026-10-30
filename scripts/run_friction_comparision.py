"""Module 1A.3 diagnostic: same cube and push, three uniform table frictions.

Run from repo root: C:\\isaacsim\\python.bat scripts\\run_friction_comparison.py
Note: This is a parameter-sensitivity experiment, NOT the localized patch test.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux, UsdPhysics

DT = 1 / 120
CUBE_SIZE = 0.05
CUBE_MASS = 0.2
PUSHER_START_X = -0.10
PUSHER_Z = 0.025
PUSH_SPEED = 0.30  # m/s; faster than 1A.2 to expose free sliding
PUSH_TRAVEL = 0.12  # m
RETRACT_SPEED = 0.30  # m/s
RETRACT_TRAVEL = 0.12  # m
COAST_S = 1.5
SETTLE_S = 1.0

# Three separated lanes share the same push and object geometry.
# Friction coefficients are dimensionless, illustrative, not calibrated.
CONDITIONS = [
    ("low", -0.20, 0.05, 0.04, (0.53, 0.72, 0.53)),
    ("nominal", 0.0, 0.45, 0.35, (0.68, 0.69, 0.73)),
    ("high", 0.20, 1.20, 1.00, (0.78, 0.62, 0.55)),
]
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True, parents=True)

world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
scene = {}
for label, y, mu_s, mu_d, color in CONDITIONS:
    material = PhysicsMaterial(
        prim_path=f"/World/Materials/{label}_surface",
        static_friction=mu_s,
        dynamic_friction=mu_d,
        restitution=0.0,
    )
    table = world.scene.add(FixedCuboid(
        prim_path=f"/World/{label}/Table", name=f"table_{label}",
        position=np.array([0, y, -0.025]), size=1,
        scale=np.array([0.8, 0.14, 0.05]), color=np.asarray(color),
    ))
    table.apply_physics_material(material)
    cube = world.scene.add(DynamicCuboid(
        prim_path=f"/World/{label}/Object", name=f"object_{label}",
        position=np.array([0, y, CUBE_SIZE / 2 + 0.001]),
        size=CUBE_SIZE, mass=CUBE_MASS,
        color=np.array([0.95, 0.25, 0.15]),
    ))
    # Keep object material the same across trials; only surface changes.
    pusher = world.scene.add(FixedCuboid(
        prim_path=f"/World/{label}/Pusher", name=f"pusher_{label}",
        position=np.array([PUSHER_START_X, y, PUSHER_Z]),
        size=1, scale=np.array([0.02, 0.10, 0.05]),
        color=np.array([0.15, 0.35, 0.95]),
    ))
    UsdPhysics.RigidBodyAPI.Apply(
        world.stage.GetPrimAtPath(f"/World/{label}/Pusher")
    ).CreateKinematicEnabledAttr(True)
    scene[label] = dict(y=y, cube=cube, pusher=pusher, mu_s=mu_s, mu_d=mu_d)

light = UsdLux.DistantLight.Define(world.stage, "/World/KeyLight")
light.CreateIntensityAttr(2000)
light.CreateAngleAttr(1)
set_camera_view(eye=np.array([0.60, -0.95, 0.75]), target=np.array([0.0, 0.0, 0.0]))

world.reset()
for _ in range(round(SETTLE_S / DT)):
    world.step(render=True)

initial = {}
for label, data in scene.items():
    p, _ = data["cube"].get_world_pose()
    initial[label] = np.array(p, dtype=float).copy()

rows = []

def record(t, phase):
    for label, data in scene.items():
        p, _ = data["cube"].get_world_pose()
        v = np.asarray(data["cube"].get_linear_velocity(), dtype=float)
        rows.append([label, data["mu_s"], data["mu_d"], phase, t,
                     *np.asarray(p, dtype=float).tolist(), *v.tolist()])

record(0.0, "settled")
step_id = 0

def move_pushers(x):
    for data in scene.values():
        data["pusher"].set_world_pose(position=np.array([x, data["y"], PUSHER_Z]))

# Driven push. "Pusher" rigid bodies are kinematic; dynamic cubes are not scripted.
num_push_steps = round(PUSH_TRAVEL / PUSH_SPEED / DT)
for i in range(1, num_push_steps + 1):
    move_pushers(PUSHER_START_X + PUSH_TRAVEL * i / num_push_steps)
    world.step(render=True)
    step_id += 1
    record(step_id * DT, "push")

# Retract smoothly at fixed speed to remove driving contact.
num_retract_steps = round(RETRACT_TRAVEL / RETRACT_SPEED / DT)
end_x = PUSHER_START_X + PUSH_TRAVEL
for i in range(1, num_retract_steps + 1):
    move_pushers(end_x - RETRACT_TRAVEL * i / num_retract_steps)
    world.step(render=True)
    step_id += 1
    record(step_id * DT, "retract")

# Coast with the pusher separated; damping and friction affect stopping.
for _ in range(round(COAST_S / DT)):
    world.step(render=True)
    step_id += 1
    record(step_id * DT, "coast")

summary = []
for label, data in scene.items():
    p, _ = data["cube"].get_world_pose()
    p = np.asarray(p, dtype=float)
    dx_mm = 1000 * (p[0] - initial[label][0])
    summary.append([label, data["mu_s"], data["mu_d"], *initial[label].tolist(), *p.tolist(), dx_mm])
    print(f"{label:8s} | static={data['mu_s']:.2f} dynamic={data['mu_d']:.2f} "
          f"| final dx={dx_mm:.3f} mm", flush=True)

with (OUT / "friction_comparison_trajectories.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["condition", "mu_static", "mu_dynamic", "phase", "time_s",
                     "object_x_m", "object_y_m", "object_z_m", "vx_m_s", "vy_m_s", "vz_m_s"])
    writer.writerows(rows)
with (OUT / "friction_comparison_summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["condition", "mu_static", "mu_dynamic", "initial_x_m", "initial_y_m",
                     "initial_z_m", "final_x_m", "final_y_m", "final_z_m", "dx_mm"])
    writer.writerows(summary)

print(f"Saved trajectories and summary to: {OUT}")
print("Close Isaac Sim window to exit.")
try:
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
