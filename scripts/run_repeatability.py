"""Module 1A.2: ten identical contact-driven pushes in Isaac Sim 5.1.

Run from repository root with Isaac Sim's bundled python.bat:
    C:\\isaacsim\\python.bat scripts\\run_repeatability.py

The dynamic cube is reset before every trial; only the pusher is scripted.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False, "renderer": "RaytracedLighting"})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, UsdLux, UsdPhysics

DT = 1.0 / 120.0
NUM_TRIALS = 10
CUBE_SIZE = 0.05
PUSH_DISTANCE = 0.10
PUSH_SPEED = 0.05
START_X = -0.10
PUSHER_Z = 0.025
SETTLE_STEPS = 120
FINAL_SETTLE_STEPS = 120
CUBE_START = np.array([0.0, 0.0, CUBE_SIZE / 2 + 0.001])
PUSHER_START = np.array([START_X, 0.0, PUSHER_Z])
IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
world.scene.add(FixedCuboid(
    prim_path="/World/Table", name="table",
    position=np.array([0.0, 0.0, -0.025]), size=1.0,
    scale=np.array([0.8, 0.6, 0.05]), color=np.array([0.45, 0.50, 0.58]),
))
cube = world.scene.add(DynamicCuboid(
    prim_path="/World/Object", name="object", position=CUBE_START,
    orientation=IDENTITY_QUAT, size=CUBE_SIZE, mass=0.2,
    color=np.array([0.95, 0.25, 0.15]),
))
pusher = world.scene.add(FixedCuboid(
    prim_path="/World/Pusher", name="pusher", position=PUSHER_START,
    size=1.0, scale=np.array([0.02, 0.10, 0.05]),
    color=np.array([0.15, 0.35, 0.95]),
))
UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath("/World/Pusher")).CreateKinematicEnabledAttr(True)

light = UsdLux.DistantLight.Define(world.stage, "/World/KeyLight")
light.CreateIntensityAttr(2000.0)
light.CreateAngleAttr(1.0)
set_camera_view(eye=np.array([0.45, -0.65, 0.45]), target=np.array([0.0, 0.0, 0.0]))

# Configure reset states once, then let world.reset() restore the cube's
# pose AND both linear/angular velocities on each trial.
cube.set_default_state(
    position=CUBE_START, orientation=IDENTITY_QUAT,
    linear_velocity=np.zeros(3), angular_velocity=np.zeros(3),
)
pusher.set_default_state(position=PUSHER_START, orientation=IDENTITY_QUAT)


def yaw_degrees(quat):
    """Yaw around Z from scalar-first (w, x, y, z) quaternion."""
    w, x, y, z = np.asarray(quat, dtype=float)
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


summary = []
trajectories = []
steps = int(round(PUSH_DISTANCE / (PUSH_SPEED * DT)))

try:
    for trial in range(1, NUM_TRIALS + 1):
        world.reset()
        for _ in range(SETTLE_STEPS):
            world.step(render=True)
        initial, _ = cube.get_world_pose()
        initial = np.asarray(initial, dtype=float).copy()
        # Read the actual settled state, not the requested spawn point.
        for step in range(steps + 1):
            x = START_X + min(PUSH_DISTANCE, step * PUSH_SPEED * DT)
            pusher.set_world_pose(position=np.array([x, 0.0, PUSHER_Z]))
            world.step(render=True)
            pos, quat = cube.get_world_pose()
            pos = np.asarray(pos, dtype=float)
            trajectories.append([trial, step * DT, x, *pos.tolist(), yaw_degrees(quat)])

        for _ in range(FINAL_SETTLE_STEPS):
            world.step(render=True)
        final, quat = cube.get_world_pose()
        final = np.asarray(final, dtype=float).copy()
        delta = (final - initial) * 1000.0
        summary.append([trial, *initial.tolist(), *final.tolist(), *delta.tolist(), yaw_degrees(quat)])
        print(f"Trial {trial:02d}/{NUM_TRIALS}: dx={delta[0]:.4f} mm, "
              f"dy={delta[1]:.4f} mm, yaw={yaw_degrees(quat):.4f} deg", flush=True)

    with (RESULTS / "repeatability_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "initial_x_m", "initial_y_m", "initial_z_m",
                         "final_x_m", "final_y_m", "final_z_m",
                         "dx_mm", "dy_mm", "dz_mm", "final_yaw_deg"])
        writer.writerows(summary)

    with (RESULTS / "repeatability_trajectories.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trial", "time_s", "pusher_x_m", "object_x_m",
                         "object_y_m", "object_z_m", "object_yaw_deg"])
        writer.writerows(trajectories)

    dx_values = np.array([r[7] for r in summary], dtype=float)
    print(f"\nMean X displacement: {dx_values.mean():.6f} mm")
    print(f"Sample std deviation: {dx_values.std(ddof=1):.6f} mm")
    print(f"Max deviation from mean: {np.max(np.abs(dx_values-dx_values.mean())):.6f} mm")
    print(f"Saved: {RESULTS / 'repeatability_summary.csv'}")
    print(f"Saved: {RESULTS / 'repeatability_trajectories.csv'}")
    print("Close Isaac Sim window to exit.")
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
