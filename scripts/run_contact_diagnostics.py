"""Module 1A.3: nominal-vs-patch contact diagnostics at 120/240 Hz.

Windows examples:
  C:\\isaacsim\\python.bat scripts\\run_contact_diagnostics.py --hz 120
  C:\\isaacsim\\python.bat scripts\\run_contact_diagnostics.py --hz 240

Retains the geometry, materials, and motion from run_friction_patch.py.
Run each frequency in a fresh Isaac Sim process.
"""
import argparse
import csv
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, choices=[120, 240], required=True)
parser.add_argument('--headless', action='store_true', help='Run without interactive viewport')
args = parser.parse_args()

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux, UsdPhysics

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
DT = 1.0 / args.hz
SIZE = 0.05
MASS = 0.2
START_X = -0.10
PUSHER_Z = 0.025
PUSHER_HALF_X = 0.01
PUSH_SPEED = 0.30
PUSH_TRAVEL = 0.12
RETRACT_SPEED = 0.30
RETRACT_TRAVEL = 0.12
SETTLE_S = 1.0
COAST_S = 1.5
PATCH_START_X = 0.040
PATCH_END_X = 0.090
LANES = [('nominal', -0.10), ('localized_patch', 0.10)]
MATERIAL_VALUES = {'nominal': (.45, .35), 'high': (1.2, 1.0)}

world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
materials = {}
for key, (mu_s, mu_d) in MATERIAL_VALUES.items():
    materials[key] = PhysicsMaterial(
        prim_path=f'/World/Materials/{key}', static_friction=mu_s,
        dynamic_friction=mu_d, restitution=0.0)

scene = {}
for label, y in LANES:
    bounds = [(-0.4, PATCH_START_X), (PATCH_START_X, PATCH_END_X),
              (PATCH_END_X, 0.4)]
    for idx, (left, right) in enumerate(bounds):
        mat_name = 'high' if label == 'localized_patch' and idx == 1 else 'nominal'
        color = (np.array([.80, .49, .29]) if mat_name == 'high'
                 else np.array([.67, .70, .74]))
        surface = world.scene.add(FixedCuboid(
            prim_path=f'/World/{label}/Surface_{idx}', name=f'surface_{label}_{idx}',
            position=np.array([(left + right) / 2, y, -0.025]), size=1,
            scale=np.array([right - left, .14, .05]), color=color))
        surface.apply_physics_material(materials[mat_name])
    cube = world.scene.add(DynamicCuboid(
        prim_path=f'/World/{label}/Object', name=f'cube_{label}',
        position=np.array([0., y, SIZE / 2 + .001]),
        size=SIZE, mass=MASS, color=np.array([.95, .25, .15])))
    pusher = world.scene.add(FixedCuboid(
        prim_path=f'/World/{label}/Pusher', name=f'pusher_{label}',
        position=np.array([START_X, y, PUSHER_Z]), size=1,
        scale=np.array([.02, .10, .05]), color=np.array([.15, .35, .95])))
    UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath(
        f'/World/{label}/Pusher')).CreateKinematicEnabledAttr(True)
    scene[label] = {'y': y, 'cube': cube, 'pusher': pusher}

light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
light.CreateIntensityAttr(2000)
light.CreateAngleAttr(1)
if not args.headless:
    set_camera_view(eye=np.array([.65, -1., .75]), target=np.array([0., 0., 0.]))

world.reset()
for _ in range(round(SETTLE_S / DT)):
    world.step(render=not args.headless)

initial = {}
for label, item in scene.items():
    p, _ = item['cube'].get_world_pose()
    initial[label] = np.asarray(p, dtype=float).copy()

rows = []
step_count = 0
commanded_pusher_x = START_X

def record(phase):
    for label, item in scene.items():
        pos, _ = item['cube'].get_world_pose()
        vel = item['cube'].get_linear_velocity()
        pusher_pos, _ = item['pusher'].get_world_pose()
        pos = np.asarray(pos, dtype=float)
        vel = np.asarray(vel, dtype=float)
        actual_x = float(pusher_pos[0])
        # X-only geometric clearance assumes yaw stays near zero; negative
        # indicates projected overlap, NOT a measured PhysX contact impulse.
        clearance = (pos[0] - SIZE / 2) - (actual_x + PUSHER_HALF_X)
        cube_left = pos[0] - SIZE / 2
        cube_right = pos[0] + SIZE / 2
        overlaps_patch = (cube_right > PATCH_START_X and cube_left < PATCH_END_X)
        rows.append([args.hz, label, phase, step_count, step_count * DT,
                     commanded_pusher_x, actual_x, pos[0], pos[1], pos[2],
                     vel[0], vel[1], vel[2], clearance * 1000,
                     1000 * (pos[0] - initial[label][0]), int(overlaps_patch)])

def move_pusher(x):
    global commanded_pusher_x
    commanded_pusher_x = float(x)
    for item in scene.values():
        item['pusher'].set_world_pose(position=np.array([
            commanded_pusher_x, item['y'], PUSHER_Z]))

record('settled')
phases = [('push', PUSH_TRAVEL, PUSH_SPEED),
          ('retract', RETRACT_TRAVEL, RETRACT_SPEED)]
for phase, travel, speed in phases:
    steps = round(travel / speed / DT)
    start = START_X if phase == 'push' else START_X + PUSH_TRAVEL
    direction = 1 if phase == 'push' else -1
    for k in range(1, steps + 1):
        move_pusher(start + direction * travel * k / steps)
        world.step(render=not args.headless)
        step_count += 1
        record(phase)

for _ in range(round(COAST_S / DT)):
    world.step(render=not args.headless)
    step_count += 1
    record('coast')

trajectory_path = OUT / f'contact_diagnostics_{args.hz}hz.csv'
fields = ['physics_hz', 'condition', 'phase', 'step', 'time_s',
          'pusher_command_x_m', 'pusher_actual_x_m', 'object_x_m',
          'object_y_m', 'object_z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
          'projected_gap_mm', 'dx_mm', 'object_overlaps_patch_x']
with trajectory_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(fields)
    writer.writerows(rows)

summary_path = OUT / f'contact_diagnostics_summary_{args.hz}hz.csv'
with summary_path.open('w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['physics_hz', 'condition', 'dx_final_mm', 'dx_end_push_mm',
                     'dx_after_push_mm', 'peak_abs_vx_m_s',
                     'min_projected_gap_mm', 'max_pusher_tracking_error_mm'])
    for label, _ in LANES:
        subset = [r for r in rows if r[1] == label]
        push_end = next(r for r in reversed(subset) if r[2] == 'push')
        final = subset[-1]
        vals = [args.hz, label, final[14], push_end[14],
                final[14] - push_end[14],
                max(abs(r[10]) for r in subset),
                min(r[13] for r in subset),
                max(abs(r[5] - r[6]) * 1000 for r in subset)]
        writer.writerow(vals)
        print(f'{args.hz}Hz {label:16s} final={vals[2]:8.3f} mm  '
              f'post-push={vals[4]:+.3f} mm  '
              f'peak|vx|={vals[5]:.3f} m/s  '
              f'min projected gap={vals[6]:+.3f} mm  '
              f'command tracking max={vals[7]:.4f} mm', flush=True)

print(f'Saved: {trajectory_path}\nSaved: {summary_path}', flush=True)
print('Geometric gap is approximate and is not a contact-force measurement.')
if not args.headless:
    print('Close Isaac Sim window to exit.')
try:
    while not args.headless and simulation_app.is_running():
        simulation_app.update()
finally:
    simulation_app.close()
