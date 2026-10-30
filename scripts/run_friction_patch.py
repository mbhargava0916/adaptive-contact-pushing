"""Module 1A.3: controlled uniform-friction vs localized-patch comparison.
Run: C:\\isaacsim\\python.bat scripts\\run_friction_patch.py
Requires Isaac Sim 5.1. Simulator integration must be tested locally.
"""
import csv
import os
from pathlib import Path

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': False, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
DT = 1 / 120
SIZE = 0.05
MASS = 0.2
START_X = -0.10
PUSHER_Z = 0.025
PUSH_SPEED = 0.30
PUSH_TRAVEL = 0.12
RETRACT_SPEED = 0.30
RETRACT_TRAVEL = 0.12
SETTLE_S = 1.0
COAST_S = 1.5
# Segment boundaries form a coplanar table. The patch is NOT an added raised slab.
PATCH_START_X = 0.020
PATCH_END_X = 0.095
LANES = [
    ('low', -0.30, 0.05, 0.04),
    ('nominal', -0.10, 0.45, 0.35),
    ('high_uniform', 0.10, 1.20, 1.00),
    ('localized_patch', 0.30, 0.45, 0.35),
]
MATERIAL_VALUES = {'low': (.05, .04), 'nominal': (.45, .35), 'high': (1.2, 1.0)}
world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
materials = {}
for key, (mu_s, mu_d) in MATERIAL_VALUES.items():
    materials[key] = PhysicsMaterial(
        prim_path=f'/World/Materials/{key}', static_friction=mu_s,
        dynamic_friction=mu_d, restitution=0.0)

scene = {}
for label, y, mu_s, mu_d in LANES:
    # All lanes use identical three-part collision geometry so any change
    # between nominal and patch lanes is only the assigned middle material.
    bounds = [(-0.4, PATCH_START_X), (PATCH_START_X, PATCH_END_X), (PATCH_END_X, 0.4)]
    for idx, (left, right) in enumerate(bounds):
        patch_region = (idx == 1)
        mat_name = 'high' if label == 'localized_patch' and patch_region else (
            'high' if label == 'high_uniform' else 'low' if label == 'low' else 'nominal')
        color = np.array([0.80, 0.49, 0.29]) if label == 'localized_patch' and patch_region else np.array([0.67, 0.70, 0.74])
        part = world.scene.add(FixedCuboid(
            prim_path=f'/World/{label}/Surface_{idx}', name=f'surface_{label}_{idx}',
            position=np.array([(left + right) / 2, y, -0.025]), size=1,
            scale=np.array([right - left, 0.14, 0.05]), color=color))
        part.apply_physics_material(materials[mat_name])
    cube = world.scene.add(DynamicCuboid(
        prim_path=f'/World/{label}/Object', name=f'cube_{label}',
        position=np.array([0, y, SIZE / 2 + 0.001]),
        size=SIZE, mass=MASS, color=np.array([0.95, .25, .15])))
    pusher = world.scene.add(FixedCuboid(
        prim_path=f'/World/{label}/Pusher', name=f'pusher_{label}',
        position=np.array([START_X, y, PUSHER_Z]), size=1,
        scale=np.array([.02, .10, .05]), color=np.array([.15, .35, .95])))
    UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(
        f'/World/{label}/Pusher')).CreateKinematicEnabledAttr(True)
    scene[label] = {'y': y, 'cube': cube, 'pusher': pusher,
                    'mu_s': mu_s, 'mu_d': mu_d}

light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
light.CreateIntensityAttr(2000)
light.CreateAngleAttr(1)
set_camera_view(eye=np.array([.65, -1.20, .90]), target=np.array([0., 0., 0.]))

world.reset()
for _ in range(round(SETTLE_S / DT)):
    world.step(render=True)

initial = {}
for label, item in scene.items():
    p, _ = item['cube'].get_world_pose()
    initial[label] = np.asarray(p, dtype=float).copy()

rows = []
step_count = 0
pusher_x = START_X

def record(phase):
    for label, item in scene.items():
        pos, _ = item['cube'].get_world_pose()
        vel = item['cube'].get_linear_velocity()
        pos, vel = np.asarray(pos, dtype=float), np.asarray(vel, dtype=float)
        rows.append([label, phase, step_count * DT, pusher_x,
                     pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                     1000 * (pos[0] - initial[label][0]),
                     PATCH_START_X <= pos[0] <= PATCH_END_X])

def move_pusher(x):
    global pusher_x
    pusher_x = float(x)
    for item in scene.values():
        item['pusher'].set_world_pose(position=np.array([pusher_x, item['y'], PUSHER_Z]))

record('settled')
phases = [('push', PUSH_TRAVEL, PUSH_SPEED), ('retract', RETRACT_TRAVEL, RETRACT_SPEED)]
for phase, travel, speed in phases:
    steps = round(travel / speed / DT)
    start = START_X if phase == 'push' else START_X + PUSH_TRAVEL
    sign = 1 if phase == 'push' else -1
    for k in range(1, steps + 1):
        move_pusher(start + sign * travel * k / steps)
        world.step(render=True)
        step_count += 1
        record(phase)

for _ in range(round(COAST_S / DT)):
    world.step(render=True)
    step_count += 1
    record('coast')

with (OUT / 'friction_patch_trajectories.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['condition', 'phase', 'time_s', 'pusher_x_m', 'object_x_m',
                'object_y_m', 'object_z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
                'dx_mm', 'center_in_patch'])
    w.writerows(rows)

with (OUT / 'friction_patch_summary.csv').open('w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['condition', 'initial_x_m', 'final_x_m', 'dx_mm', 'final_vx_m_s'])
    for label, item in scene.items():
        pos, _ = item['cube'].get_world_pose()
        vel = item['cube'].get_linear_velocity()
        dx = 1000 * (pos[0] - initial[label][0])
        print(f'{label:16s} final dx={dx:9.3f} mm; final vx={vel[0]:+.6f} m/s', flush=True)
        w.writerow([label, initial[label][0], pos[0], dx, vel[0]])

print('Patch interval (world X):', PATCH_START_X, 'to', PATCH_END_X, 'm')
print('Push ends at', PUSH_TRAVEL / PUSH_SPEED, 's; retract ends at',
      PUSH_TRAVEL / PUSH_SPEED + RETRACT_TRAVEL / RETRACT_SPEED, 's')
print('Results:', OUT)
print('Close Isaac Sim window to exit.')
try:
    while simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
