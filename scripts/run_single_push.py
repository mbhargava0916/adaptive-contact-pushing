"""Module 1A.1: first contact-driven push in Isaac Sim 5.1.

Run with Isaac Sim's bundled python.bat (Windows), not system Python.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RaytracedLighting",
    "display_options": 3286,
    "width": 1280,
    "height": 720,
})

# Isaac Sim / Omniverse imports must follow SimulationApp initialization.
import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdPhysics, UsdLux, Gf

DT = 1.0 / 120.0
CUBE_SIZE = 0.05
PUSH_DISTANCE = 0.10
PUSH_SPEED = 0.05
START_X = -0.10
TABLE_TOP_Z = 0.0
PUSHER_Z = 0.025

world = World(physics_dt=DT, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
world.scene.add(
    FixedCuboid(
        prim_path="/World/Table",
        name="table",
        position=np.array([0.0, 0.0, -0.025]),
        size=1.0,
        scale=np.array([0.8, 0.6, 0.05]),
        color=np.array([0.45, 0.50, 0.58]),
    )
)
cube = world.scene.add(
    DynamicCuboid(
        prim_path="/World/Object",
        name="object",
        position=np.array([0.0, 0.0, CUBE_SIZE / 2.0 + 0.001]),
        size=CUBE_SIZE,
        mass=0.2,
        color=np.array([0.95, 0.25, 0.15]),
    )
)
pusher = world.scene.add(
    FixedCuboid(
        prim_path="/World/Pusher",
        name="pusher",
        position=np.array([START_X, 0.0, PUSHER_Z]),
        size=1.0,
        scale=np.array([0.02, 0.10, 0.05]),
        color=np.array([0.15, 0.35, 0.95]),
    )
)
# Treat the externally prescribed moving pusher as a kinematic rigid body.
# Its pose is scripted; the target object's pose is never scripted.
stage = world.stage
pusher_prim = stage.GetPrimAtPath("/World/Pusher")
rigid_api = UsdPhysics.RigidBodyAPI.Apply(pusher_prim)
rigid_api.CreateKinematicEnabledAttr(True)

# Position the viewport camera above and beside the table.
set_camera_view(
    eye=np.array([0.45, -0.65, 0.45]),
    target=np.array([0.0, 0.0, 0.0]),
)

# Illuminate the scene.
light = UsdLux.DistantLight.Define(
    world.stage,
    "/World/KeyLight",
)
light.CreateIntensityAttr(2000.0)
light.CreateAngleAttr(1.0)

world.reset()
for _ in range(120):
    world.step(render=True)

initial_position, _ = cube.get_world_pose()
initial_position = np.asarray(initial_position, dtype=float).copy()
rows = []

# Move pusher at a fixed speed over exactly 100 mm.
num_steps = int(round(PUSH_DISTANCE / (PUSH_SPEED * DT)))
for i in range(num_steps + 1):
    x = START_X + min(PUSH_DISTANCE, i * PUSH_SPEED * DT)
    pusher.set_world_pose(position=np.array([x, 0.0, PUSHER_Z]))
    world.step(render=True)
    pos, _ = cube.get_world_pose()
    rows.append([i * DT, x, *np.asarray(pos, dtype=float).tolist()])

# Let the cube settle before measuring the final position.
for _ in range(120):
    world.step(render=True)

final_position, _ = cube.get_world_pose()
final_position = np.asarray(final_position, dtype=float).copy()

output_path = Path(__file__).resolve().parents[1] / "results" / "first_push_trajectory.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["time_s", "pusher_x_m", "object_x_m", "object_y_m", "object_z_m"])
    writer.writerows(rows)

print("Initial cube position (m):", initial_position)
print("Final cube position (m):  ", final_position)
print("Cube displacement (mm):  ", (final_position - initial_position) * 1000.0)
print("Trajectory CSV:           ", output_path)
print("Test complete. Close the Isaac Sim window to exit.")

try:
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()

