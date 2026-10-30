"""Module 1A.3c: isolate segmented-table seams from localized friction.

Run with Isaac Sim 5.1's Python, from the project root:
  C:\\isaacsim\\python.bat scripts\\run_localized_patch_clockfix.py --hz 240
  C:\\isaacsim\\python.bat scripts\\run_localized_patch_clockfix.py --hz 120

A: one continuous nominal table
B: three coplanar nominal segments
C: three coplanar segments, high-friction middle segment

All cubes are released at the SAME velocity (there is no pusher in this test).
Captures pitch, angular velocity, height, and X displacement. A footprint
intersection is only a geometric estimate, not a collision contact report.
"""
import argparse
import csv
import math
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Test whether table seams cause cube tipping')
parser.add_argument('--hz', type=int, choices=[120, 240, 480], default=240)
parser.add_argument('--headless', action='store_true')
parser.add_argument('--launch-speed', type=float, default=0.70)
args = parser.parse_args()
if args.launch_speed <= 0:
    parser.error('--launch-speed must be greater than zero')

os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
simulation_app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.viewports import set_camera_view
from pxr import UsdLux

DT = 1.0 / args.hz
CUBE_SIZE = 0.050
CUBE_MASS = 0.20
PATCH_START_X, PATCH_END_X = 0.050, 0.150
TABLE_X_MIN, TABLE_X_MAX = -0.35, 0.65
TABLE_HEIGHT, TABLE_WIDTH = 0.05, 0.18
SEGMENTS = [(TABLE_X_MIN, PATCH_START_X),
            (PATCH_START_X, PATCH_END_X),
            (PATCH_END_X, TABLE_X_MAX)]
CONDITIONS = [('continuous_nominal', -0.25),
              ('segmented_nominal', 0.0),
              ('segmented_patch', 0.25)]
SETTLE_S, OBSERVE_S = 0.7, 2.0
STOP_EPS = 0.005
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)


def pitch_from_wxyz(quat):
    """XYZ-convention pitch about Y, expressed in degrees (valid near upright)."""
    w, x, y, z = [float(v) for v in quat]
    return math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (w*y - z*x)))))


try:
    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    nominal = PhysicsMaterial('/World/Materials/Nominal', static_friction=0.45,
                              dynamic_friction=0.35, restitution=0.0)
    high = PhysicsMaterial('/World/Materials/High', static_friction=1.20,
                           dynamic_friction=1.00, restitution=0.0)
    cube_material = PhysicsMaterial('/World/Materials/Cube', static_friction=0.45,
                                    dynamic_friction=0.35, restitution=0.0)

    cubes = {}
    for condition, lane_y in CONDITIONS:
        spans = [(TABLE_X_MIN, TABLE_X_MAX)] if condition == 'continuous_nominal' else SEGMENTS
        for idx, (x0, x1) in enumerate(spans):
            patch_segment = condition == 'segmented_patch' and idx == 1
            surface = world.scene.add(FixedCuboid(
                prim_path=f'/World/{condition}/Surface_{idx}',
                name=f'{condition}_surface_{idx}',
                position=np.array([(x0+x1)/2, lane_y, -TABLE_HEIGHT/2]),
                size=1.0, scale=np.array([x1-x0, TABLE_WIDTH, TABLE_HEIGHT]),
                color=np.array([0.85, 0.49, 0.24] if patch_segment
                               else [0.60, 0.65, 0.70])))
            surface.apply_physics_material(high if patch_segment else nominal)

        cube = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{condition}/Cube', name=f'{condition}_cube',
            position=np.array([0.0, lane_y, 0.026]),
            size=CUBE_SIZE, mass=CUBE_MASS,
            color=np.array([0.9, 0.22, 0.18])))
        cube.apply_physics_material(cube_material)
        cubes[condition] = cube

    light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
    light.CreateIntensityAttr(2000.0)
    light.CreateAngleAttr(1.0)
    if not args.headless:
        set_camera_view(eye=np.array([0.42, -1.10, 0.75]),
                        target=np.array([0.10, 0.0, 0.0]))

    world.reset()
    for _ in range(round(SETTLE_S/DT)):
        world.step(render=not args.headless)

    starts = {}
    for condition, cube in cubes.items():
        pos, _ = cube.get_world_pose()
        starts[condition] = np.array(pos, dtype=float)
        cube.set_angular_velocity(np.zeros(3))
        cube.set_linear_velocity(np.array([args.launch_speed, 0.0, 0.0]))

    records = []
    def sample(k):
        t = float(world.current_time) - clock0
        for condition, cube in cubes.items():
            pos, quat = cube.get_world_pose()
            v = cube.get_linear_velocity()
            omega = cube.get_angular_velocity()
            pos = np.asarray(pos, dtype=float)
            v = np.asarray(v, dtype=float)
            omega = np.asarray(omega, dtype=float)
            # Approximation valid only if rotation about Y remains small.
            overlap = int(pos[0] + CUBE_SIZE/2 > PATCH_START_X and
                          pos[0] - CUBE_SIZE/2 < PATCH_END_X)
            records.append(dict(
                physics_hz=args.hz, condition=condition, step=k, time_s=t,
                object_x_m=pos[0], object_y_m=pos[1], object_z_m=pos[2],
                dx_mm=1000*(pos[0]-starts[condition][0]),
                dz_mm=1000*(pos[2]-starts[condition][2]),
                vx_m_s=v[0], vy_m_s=v[1], vz_m_s=v[2],
                pitch_deg=pitch_from_wxyz(quat),
                omega_x_rad_s=omega[0], omega_y_rad_s=omega[1],
                omega_z_rad_s=omega[2],
                projected_patch_overlap=overlap))

    clock0 = float(world.current_time)
    sample(0)
    for step in range(1, round(OBSERVE_S/DT) + 1):
        world.step(render=not args.headless)
        sample(step)

    elapsed = float(world.current_time) - clock0
    expected = round(OBSERVE_S/DT)*DT
    print(f'World clock elapsed: {elapsed:.6f} s; expected: {expected:.6f} s; '
          f'difference: {elapsed-expected:+.6f} s')

    trajectory_path = OUT / f'seam_diagnostics_{args.hz}hz.csv'
    with trajectory_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    summary_path = OUT / f'seam_diagnostics_summary_{args.hz}hz.csv'
    summary_fields = ['physics_hz', 'condition', 'launch_speed_m_s',
                      'final_dx_mm', 'max_abs_pitch_deg', 'max_abs_omega_y_rad_s',
                      'max_height_rise_mm', 'min_vx_m_s',
                      'first_projected_patch_overlap_s', 'last_moving_s']
    summaries = {}
    with summary_path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=summary_fields)
        writer.writeheader()
        for condition, _ in CONDITIONS:
            data = [row for row in records if row['condition'] == condition]
            first_overlap = next((r['time_s'] for r in data
                                  if r['projected_patch_overlap']), None)
            last_moving = max((r['time_s'] for r in data
                               if abs(r['vx_m_s']) > STOP_EPS), default=None)
            result = dict(physics_hz=args.hz, condition=condition,
                          launch_speed_m_s=args.launch_speed,
                          final_dx_mm=data[-1]['dx_mm'],
                          max_abs_pitch_deg=max(abs(r['pitch_deg']) for r in data),
                          max_abs_omega_y_rad_s=max(abs(r['omega_y_rad_s']) for r in data),
                          max_height_rise_mm=max(r['dz_mm'] for r in data),
                          min_vx_m_s=min(r['vx_m_s'] for r in data),
                          first_projected_patch_overlap_s=first_overlap,
                          last_moving_s=last_moving)
            writer.writerow(result)
            summaries[condition] = result
            print(f"{args.hz} Hz {condition:19s} "
                  f"dx={result['final_dx_mm']:8.3f} mm  "
                  f"max|pitch|={result['max_abs_pitch_deg']:7.3f} deg  "
                  f"max rise={result['max_height_rise_mm']:7.3f} mm  "
                  f"min vx={result['min_vx_m_s']:+.3f} m/s")

    baseline = summaries['continuous_nominal']['final_dx_mm']
    print(f"Segmented nominal - continuous: "
          f"{summaries['segmented_nominal']['final_dx_mm'] - baseline:+.3f} mm")
    print(f"Patch - segmented nominal: "
          f"{summaries['segmented_patch']['final_dx_mm'] - summaries['segmented_nominal']['final_dx_mm']:+.3f} mm")
    print(f'Saved: {trajectory_path}\nSaved: {summary_path}')
    print('Pitch is Euler XYZ Y-angle; geometric patch overlap is NOT a contact-force reading.')
    print('No pushing force is applied: all cubes start with assigned linear velocity.')
    if not args.headless:
        print('Close Isaac Sim window to exit.')
        while simulation_app.is_running():
            simulation_app.update()
finally:
    simulation_app.close()
