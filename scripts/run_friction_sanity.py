"""Module 1A.3c: single-surface sliding sanity test.

Run (from repo root):
  C:\\isaacsim\\python.bat scripts\\run_release_friction_validation.py --hz 120
  C:\\isaacsim\\python.bat scripts\\run_release_friction_validation.py --hz 240

The cube is given the same initial horizontal velocity in both lanes AFTER
settling. This is a diagnostic control, NOT a physical pusher or impulse model.
Contact with a patch is approximated by the cube's X footprint; contact-force
reports and a genuine push/release actuator are separate future validations.
"""
import argparse
import csv
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, choices=[120, 240, 480], required=True)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--launch-speed', type=float, default=0.70,
                    help='Initial X velocity in m/s; both lanes get same value')
args = parser.parse_args()
if args.launch_speed <= 0:
    parser.error('--launch-speed must be positive')

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
OUT.mkdir(parents=True, exist_ok=True)
DT = 1.0 / args.hz
CUBE_SIZE = 0.05
MASS = 0.2
PATCH_START_X = 0.075  # 50 mm ahead of the initially centered cube's front edge
PATCH_END_X = 0.20
TABLE_LEFT = -0.35
TABLE_RIGHT = 0.65
SETTLE_S = 0.7
OBSERVE_S = 2.0
SPEED_EPS = 0.005
LANES = [('frictionless', -0.22), ('low', -0.07), ('nominal', 0.08), ('high', 0.23)]

try:
    world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
    # IMPORTANT: One continuous collider per lane (no internal table seams).
    # Same cube material for every condition, explicitly assigned.
    mats = {
        'frictionless': PhysicsMaterial('/World/Materials/frictionless', static_friction=0.0,
                                        dynamic_friction=0.0, restitution=0.0),
        'low': PhysicsMaterial('/World/Materials/low', static_friction=0.05,
                               dynamic_friction=0.04, restitution=0.0),
        'nominal': PhysicsMaterial('/World/Materials/nominal', static_friction=0.45,
                                   dynamic_friction=0.35, restitution=0.0),
        'high': PhysicsMaterial('/World/Materials/high', static_friction=1.20,
                                dynamic_friction=1.00, restitution=0.0),
    }
    cube_mat = PhysicsMaterial('/World/Materials/cube', static_friction=0.45,
                                dynamic_friction=0.35, restitution=0.0)
    scene = {}
    for label, y in LANES:
        surface = world.scene.add(FixedCuboid(
            prim_path=f'/World/{label}/Surface',
            name=f'{label}_surface',
            position=np.array([0.85, y, -0.025]),
            size=1.0, scale=np.array([2.50, 0.13, 0.05]),
            color=np.array([0.60, 0.65, 0.70])))
        surface.apply_physics_material(mats[label])
        cube = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{label}/Cube', name=f'{label}_cube',
            position=np.array([0.0, y, 0.026]), size=CUBE_SIZE,
            mass=MASS, color=np.array([0.90, 0.22, 0.18])))
        cube.apply_physics_material(cube_mat)
        scene[label] = cube

    light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
    light.CreateIntensityAttr(2000.0)
    light.CreateAngleAttr(1.0)
    if not args.headless:
        set_camera_view(eye=np.array([0.55, -0.95, 0.7]),
                        target=np.array([0.10, 0.0, 0.0]))

    world.reset()
    for _ in range(round(SETTLE_S / DT)):
        world.step(render=not args.headless)

    starts = {}
    for label, cube in scene.items():
        position, _ = cube.get_world_pose()
        starts[label] = float(position[0])
        cube.set_linear_velocity(np.array([args.launch_speed, 0.0, 0.0]))
        cube.set_angular_velocity(np.array([0.0, 0.0, 0.0]))

    rows = []
    def record(step):
        t = step * DT
        for label, cube in scene.items():
            position, _ = cube.get_world_pose()
            velocity = cube.get_linear_velocity()
            position = np.asarray(position, dtype=float)
            velocity = np.asarray(velocity, dtype=float)
            lo, hi = float(position[0] - CUBE_SIZE/2), float(position[0] + CUBE_SIZE/2)
            projected_patch_overlap = 0  # No patch in this isolated sanity test
            rows.append([args.hz, label, step, t, *position, *velocity,
                         1000*(float(position[0])-starts[label]),
                         projected_patch_overlap])

    record(0)
    for k in range(1, round(OBSERVE_S / DT) + 1):
        world.step(render=not args.headless)
        record(k)

    header = ['physics_hz','condition','step','time_s','object_x_m','object_y_m',
              'object_z_m','vx_m_s','vy_m_s','vz_m_s','dx_mm','projected_patch_overlap']
    trajectory_path = OUT / f'friction_sanity_{args.hz}hz.csv'
    with trajectory_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    summary_header = ['physics_hz','condition','launch_speed_m_s','final_dx_mm',
                      'first_patch_overlap_time_s','last_moving_time_s',
                      'final_abs_vx_m_s','stopped_by_end']
    summary_path = OUT / f'friction_sanity_summary_{args.hz}hz.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(summary_header)
        for label, _ in LANES:
            r = [a for a in rows if a[1] == label]
            first_overlap = next((a[3] for a in r if a[-1]), None)
            last_moving = max((a[3] for a in r if abs(a[7]) > SPEED_EPS), default=None)
            final_vx = abs(r[-1][7])
            stopped = final_vx <= SPEED_EPS
            writer.writerow([args.hz,label,args.launch_speed,r[-1][10],
                             first_overlap,last_moving,final_vx,int(stopped)])
            print(f'{args.hz} Hz {label:16s} displacement={r[-1][10]:8.3f} mm  '
                  f'first patch overlap={first_overlap} s  '
                  f'last moving={last_moving} s  final |vx|={final_vx:.5f} m/s',flush=True)
    print(f'Saved: {trajectory_path}\nSaved: {summary_path}',flush=True)
    print('IMPORTANT: Velocity is assigned directly after settling (friction-only control).')
    print('All lanes use a single continuous table collider with an explicit cube material.')
    print('Friction values are material inputs; effective coefficients depend on contact pairing.')
    if not args.headless:
        print('Close the Isaac Sim window to exit.')
        while simulation_app.is_running():
            simulation_app.update()
finally:
    simulation_app.close()
