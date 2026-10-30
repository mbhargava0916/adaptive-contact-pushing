"""Module 1A.3b: nominal vs localized-patch pushing, smooth command and Hz sweep.

Run separately (fresh Isaac Sim process for each):
    C:\\isaacsim\\python.bat scripts\\run_smooth_contact_validation.py --hz 120
    C:\\isaacsim\\python.bat scripts\\run_smooth_contact_validation.py --hz 240
Optionally: --hz 480; add --headless for faster execution.

Two lanes use the same dynamic cube, pusher and commanded motion.
Only the table collision material changes within the patch interval.
"""
import argparse
import csv
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, choices=(120, 240, 480), required=True)
parser.add_argument('--headless', action='store_true')
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
PUSHER_Z = SIZE / 2
PUSHER_HALF_X = 0.01
PUSH_TRAVEL = 0.12
PUSH_DURATION = 0.65   # smoothstep: peak velocity ~= 0.277 m/s
RETRACT_DURATION = 0.45
COAST_DURATION = 1.50
SETTLE_DURATION = 1.0
PATCH_START_X = 0.040  # cube initially extends to +0.025 m: 15 mm of clearance
PATCH_END_X = 0.090
LANES = [('nominal', -0.10), ('localized_patch', 0.10)]
MATERIAL_VALUES = {'nominal': (0.45, 0.35), 'high': (1.20, 1.00)}


def smoothstep(u):
    u = float(np.clip(u, 0., 1.))
    return u * u * (3. - 2. * u)


def phase_steps(duration):
    return round(duration / DT)


try:
    world = World(physics_dt=DT, rendering_dt=1 / 60, stage_units_in_meters=1.0)
    materials = {}
    for key, (mu_s, mu_d) in MATERIAL_VALUES.items():
        materials[key] = PhysicsMaterial(
            prim_path=f'/World/Materials/{key}', static_friction=mu_s,
            dynamic_friction=mu_d, restitution=0.0)

    scene = {}
    for label, y in LANES:
        segments = [(-0.40, PATCH_START_X),
                    (PATCH_START_X, PATCH_END_X),
                    (PATCH_END_X, 0.40)]
        for idx, (left, right) in enumerate(segments):
            mat_name = ('high' if label == 'localized_patch' and idx == 1
                        else 'nominal')
            color = (np.array([.80, .49, .29]) if mat_name == 'high'
                     else np.array([.67, .70, .74]))
            surface = world.scene.add(FixedCuboid(
                prim_path=f'/World/{label}/Surface_{idx}',
                name=f'surface_{label}_{idx}',
                position=np.array([(left + right) / 2, y, -0.025]),
                size=1,
                scale=np.array([right - left, .14, .05]),
                color=color))
            surface.apply_physics_material(materials[mat_name])

        cube = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{label}/Object', name=f'cube_{label}',
            position=np.array([0., y, SIZE / 2 + .001]), size=SIZE,
            mass=MASS, color=np.array([.95, .25, .15])))
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
        set_camera_view(eye=np.array([.65, -1., .75]),
                        target=np.array([0., 0., 0.]))

    world.reset()
    for _ in range(phase_steps(SETTLE_DURATION)):
        world.step(render=not args.headless)

    initial = {}
    for label, item in scene.items():
        p, _ = item['cube'].get_world_pose()
        initial[label] = float(p[0])

    rows = []
    step_count = 0
    commanded_x = START_X
    current_phase = 'settled'

    def record():
        for label, item in scene.items():
            pos, _ = item['cube'].get_world_pose()
            vel = item['cube'].get_linear_velocity()
            pusher_pos, _ = item['pusher'].get_world_pose()
            pos = np.asarray(pos, dtype=float)
            vel = np.asarray(vel, dtype=float)
            actual_x = float(pusher_pos[0])
            cube_left = pos[0] - SIZE / 2
            cube_right = pos[0] + SIZE / 2
            gap_mm = 1000 * (cube_left - (actual_x + PUSHER_HALF_X))
            patch_overlap = cube_right > PATCH_START_X and cube_left < PATCH_END_X
            rows.append([args.hz, label, current_phase, step_count,
                         step_count * DT, commanded_x, actual_x,
                         pos[0], pos[1], pos[2], vel[0], vel[1], vel[2],
                         1000 * (pos[0] - initial[label]), gap_mm,
                         int(patch_overlap)])

    def move_and_step(target_x):
        global commanded_x, step_count
        commanded_x = float(target_x)
        for item in scene.values():
            item['pusher'].set_world_pose(position=np.array([
                commanded_x, item['y'], PUSHER_Z]))
        world.step(render=not args.headless)
        step_count += 1
        record()

    record()
    current_phase = 'push'
    for k in range(1, phase_steps(PUSH_DURATION) + 1):
        u = k / phase_steps(PUSH_DURATION)
        move_and_step(START_X + PUSH_TRAVEL * smoothstep(u))

    current_phase = 'retract'
    for k in range(1, phase_steps(RETRACT_DURATION) + 1):
        u = k / phase_steps(RETRACT_DURATION)
        move_and_step(START_X + PUSH_TRAVEL * (1. - smoothstep(u)))

    current_phase = 'coast'
    for _ in range(phase_steps(COAST_DURATION)):
        world.step(render=not args.headless)
        step_count += 1
        record()

    header = ['physics_hz', 'condition', 'phase', 'step', 'time_s',
              'pusher_command_x_m', 'pusher_actual_x_m', 'object_x_m',
              'object_y_m', 'object_z_m', 'vx_m_s', 'vy_m_s', 'vz_m_s',
              'dx_mm', 'projected_gap_mm', 'overlaps_patch_x']
    traj_path = OUT / f'smooth_contact_{args.hz}hz.csv'
    with traj_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    summary_path = OUT / f'smooth_contact_summary_{args.hz}hz.csv'
    with summary_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['physics_hz', 'condition', 'final_dx_mm',
                         'end_push_dx_mm', 'post_push_dx_mm', 'peak_abs_vx_m_s',
                         'min_projected_gap_mm', 'first_patch_overlap_time_s',
                         'max_pusher_tracking_error_mm'])
        for label, _ in LANES:
            subset = [r for r in rows if r[1] == label]
            end_push = next(r for r in reversed(subset) if r[2] == 'push')
            final = subset[-1]
            first_overlap = next((r[4] for r in subset if r[15]), None)
            result = [args.hz, label, final[13], end_push[13],
                      final[13] - end_push[13],
                      max(abs(r[10]) for r in subset),
                      min(r[14] for r in subset), first_overlap,
                      max(abs(r[5] - r[6]) * 1000 for r in subset)]
            writer.writerow(result)
            print(f'{args.hz} Hz {label:16s} final={result[2]:8.3f} mm  '
                  f'post-push={result[4]:+.3f} mm  '
                  f'peak|vx|={result[5]:.3f} m/s  '
                  f'first overlap={first_overlap} s  '
                  f'min gap={result[6]:+.3f} mm', flush=True)

    print(f'Saved: {traj_path}\nSaved: {summary_path}', flush=True)
    print('NOTE: Patch overlap is an X-footprint approximation, not confirmed force/contact.')
    print('NOTE: Smoothstep changes the commanded motion, but still sets kinematic poses each step.')
    if not args.headless:
        print('Close Isaac Sim window to exit.')
        while simulation_app.is_running():
            simulation_app.update()
finally:
    simulation_app.close()
