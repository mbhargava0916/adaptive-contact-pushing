"""Module 1A.3d refined: contact-driven push, release, THEN patch encounter.

Isaac Sim 5.1 standalone:
  C:\\isaacsim\\python.bat scripts\\run_push_release_mesh.py --hz 240
  C:\\isaacsim\\python.bat scripts\\run_push_release_mesh.py --hz 120

Cube velocity is NEVER prescribed. A kinematic pusher contacts the cube.
The push has a short acceleration ramp followed by constant velocity, so the
cube can retain velocity at withdrawal. The pusher retracts; the cube coasts.

Note: Pose-stepped kinematic pushers may still cause solver artifacts. This is
an integration experiment, not an experimentally validated force controller.
"""
import argparse
import csv
import math
import os
import traceback
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--hz', type=int, choices=(120, 240, 480), default=240)
parser.add_argument('--headless', action='store_true')
args = parser.parse_args()
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
app = SimulationApp({'headless': args.headless, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, FixedCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Vt, UsdGeom, UsdPhysics, UsdShade, UsdLux

DT = 1.0 / args.hz
CUBE_SIZE, CUBE_MASS = 0.05, 0.2
TABLE_THICKNESS, HALF_WIDTH = 0.05, 0.09
PATCH_BEGIN, PATCH_END = 0.105, 0.205
# Table is a single, watertight triangle mesh, subdivided at patch boundaries.
X_POINTS = [-0.35, PATCH_BEGIN, PATCH_END, 0.65]
LANES = [('nominal', -0.13), ('patch', 0.13)]
PUSHER_START_X = -0.09
PUSHER_HALF_LENGTH = 0.01
PUSHER_Z = 0.025
PUSH_TRAVEL = 0.095
PUSH_DURATION = 0.18
RAMP_DURATION = 0.035
RETRACT_DURATION = 0.14
SETTLE_DURATION = 0.65
COAST_DURATION = 1.5
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)


def pitch_deg(q):
    w, x, y, z = map(float, q)
    return math.degrees(math.asin(max(-1.0, min(1.0, 2*(w*y-z*x)))))


def build_table(stage, path, lane_y, nominal_path, patch_path, patch_enabled):
    mesh = UsdGeom.Mesh.Define(stage, path)
    pts = [Gf.Vec3f(x, lane_y + side * HALF_WIDTH, z)
           for z in (0.0, -TABLE_THICKNESS)
           for x in X_POINTS for side in (-1, 1)]
    def ix(layer, j, side):
        return layer * 8 + 2*j + side
    triangles, patch_faces = [], []
    def tri(a, b, c, on_patch=False):
        if on_patch:
            patch_faces.append(len(triangles))
        triangles.append((a, b, c))
    for j in range(3):
        v00, v01 = ix(0,j,0), ix(0,j,1)
        v10, v11 = ix(0,j+1,0), ix(0,j+1,1)
        b00, b01 = ix(1,j,0), ix(1,j,1)
        b10, b11 = ix(1,j+1,0), ix(1,j+1,1)
        tri(v00,v10,v11,j==1); tri(v00,v11,v01,j==1)
        tri(b00,b11,b10); tri(b00,b01,b11)
        tri(v00,b00,b10); tri(v00,b10,v10)
        tri(v01,v11,b11); tri(v01,b11,b01)
    for j in (0,3):
        va,vb = ix(0,j,0), ix(0,j,1)
        ba,bb = ix(1,j,0), ix(1,j,1)
        if j==0:
            tri(va,vb,bb); tri(va,bb,ba)
        else:
            tri(va,bb,vb); tri(va,ba,bb)
    mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(triangles)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([v for f in triangles for v in f]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    collision.CreateApproximationAttr().Set('none')
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    binding.Bind(UsdShade.Material.Get(stage, nominal_path),
                 UsdShade.Tokens.weakerThanDescendants, 'physics')
    if patch_enabled:
        subset = binding.CreateMaterialBindSubset('HighFrictionTopFaces',
                                                  Vt.IntArray(patch_faces))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            UsdShade.Material.Get(stage, patch_path),
            UsdShade.Tokens.weakerThanDescendants, 'physics')
        print(f'{path}: high-friction triangles {patch_faces}')
    return mesh


def push_position(t):
    """Smooth initial acceleration, then constant speed; no artificial braking."""
    ramp = RAMP_DURATION
    speed = PUSH_TRAVEL / (PUSH_DURATION - 0.5*ramp)
    if t <= ramp:
        return PUSHER_START_X + 0.5*speed*t*t/ramp
    return PUSHER_START_X + speed*(t - 0.5*ramp)


try:
    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    nominal = PhysicsMaterial('/World/Materials/Nominal', static_friction=0.45,
                              dynamic_friction=0.35, restitution=0.0)
    high = PhysicsMaterial('/World/Materials/High', static_friction=1.20,
                           dynamic_friction=1.0, restitution=0.0)
    cube_mat = PhysicsMaterial('/World/Materials/Cube', static_friction=0.45,
                               dynamic_friction=0.35, restitution=0.0)
    pusher_mat = PhysicsMaterial('/World/Materials/Pusher', static_friction=0.5,
                                 dynamic_friction=0.4, restitution=0.0)
    lanes = {}
    for label, y in LANES:
        build_table(world.stage, f'/World/{label}/Table', y,
                    '/World/Materials/Nominal', '/World/Materials/High', label=='patch')
        cube = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{label}/Cube', name=f'{label}_cube',
            position=np.array([0., y, CUBE_SIZE/2 + 0.001]), size=CUBE_SIZE,
            mass=CUBE_MASS, color=np.array([0.9, 0.23, 0.15])))
        cube.apply_physics_material(cube_mat)
        pusher = world.scene.add(FixedCuboid(
            prim_path=f'/World/{label}/Pusher', name=f'{label}_pusher',
            position=np.array([PUSHER_START_X, y, PUSHER_Z]), size=1.0,
            scale=np.array([2*PUSHER_HALF_LENGTH, 0.10, 0.05]),
            color=np.array([0.18, 0.40, 0.95])))
        pusher.apply_physics_material(pusher_mat)
        UsdPhysics.RigidBodyAPI.Apply(
            world.stage.GetPrimAtPath(f'/World/{label}/Pusher')
        ).CreateKinematicEnabledAttr(True)
        lanes[label] = dict(y=y, cube=cube, pusher=pusher)

    light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
    light.CreateIntensityAttr(2000.)
    if not args.headless:
        set_camera_view(eye=np.array([0.52, -0.85, 0.56]),
                        target=np.array([0.07, 0., 0.]))
    world.reset()
    for _ in range(round(SETTLE_DURATION/DT)):
        world.step(render=not args.headless)

    starts = {name: float(v['cube'].get_world_pose()[0][0])
              for name,v in lanes.items()}
    rows = []
    t0 = float(world.current_time)
    phase = 'initial'
    command_x = PUSHER_START_X
    step_id = 0

    def sample():
        t = float(world.current_time) - t0
        for label, data in lanes.items():
            p, q = data['cube'].get_world_pose()
            p = np.asarray(p, dtype=float)
            linear = np.asarray(data['cube'].get_linear_velocity(), dtype=float)
            angular = np.asarray(data['cube'].get_angular_velocity(), dtype=float)
            pp, _ = data['pusher'].get_world_pose()
            # Approximate projected gap; assumes a cube with small pitch.
            gap = float(p[0] - CUBE_SIZE/2 - (pp[0] + PUSHER_HALF_LENGTH))
            overlap = bool(p[0]+CUBE_SIZE/2 > PATCH_BEGIN and
                           p[0]-CUBE_SIZE/2 < PATCH_END)
            rows.append(dict(hz=args.hz, lane=label, phase=phase,
                             step=step_id, time_s=t, cube_x_m=float(p[0]),
                             cube_y_m=float(p[1]), cube_z_m=float(p[2]),
                             displacement_mm=(float(p[0])-starts[label])*1000,
                             vx_m_s=float(linear[0]), vz_m_s=float(linear[2]),
                             pitch_deg=pitch_deg(q), omega_y_rad_s=float(angular[1]),
                             pusher_x_m=float(pp[0]), command_x_m=command_x,
                             projected_gap_mm=gap*1000,
                             footprint_patch_overlap=int(overlap)))

    def advance(new_x):
        global command_x, step_id
        command_x = float(new_x)
        for data in lanes.values():
            data['pusher'].set_world_pose(position=np.array([
                command_x, data['y'], PUSHER_Z]))
        world.step(render=not args.headless)
        step_id += 1
        sample()

    sample()
    phase = 'push'
    push_n = round(PUSH_DURATION/DT)
    for k in range(1, push_n+1):
        advance(push_position(PUSH_DURATION * k/push_n))
    push_end_idx = len(rows)
    phase = 'retract'
    end_x = push_position(PUSH_DURATION)
    retract_n = round(RETRACT_DURATION/DT)
    for k in range(1, retract_n+1):
        u = k/retract_n
        s = 3*u*u-2*u*u*u
        advance(end_x + (PUSHER_START_X-end_x)*s)
    phase = 'coast'
    for k in range(round(COAST_DURATION/DT)):
        world.step(render=not args.headless)
        step_id += 1
        sample()

    elapsed = float(world.current_time)-t0
    expected_steps=push_n+retract_n+round(COAST_DURATION/DT)
    print(f'World clock elapsed: {elapsed:.6f} s; '
          f'expected executed steps time: {expected_steps*DT:.6f} s; '
          f'requested phases sum: {PUSH_DURATION+RETRACT_DURATION+COAST_DURATION:.6f} s')
    traj = OUT / f'push_release_mesh_refined_{args.hz}hz.csv'
    with traj.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    summary=OUT / f'push_release_mesh_refined_summary_{args.hz}hz.csv'
    columns=['hz','lane','end_push_dx_mm','end_push_vx_m_s',
             'first_separated_time_s','first_separated_vx_m_s','final_dx_mm',
             'post_push_dx_mm','peak_abs_pitch_deg','max_z_rise_mm',
             'first_patch_overlap_time_s','min_gap_mm',
             'separation_dx_mm','post_separation_dx_mm','patch_after_separation',
             'separation_to_patch_s','stop_time_s']
    with summary.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader()
        for name,_ in LANES:
            subset=[r for r in rows if r['lane']==name]
            push_end=next(r for r in reversed(subset) if r['phase']=='push')
            final=subset[-1]
            separated=next((r for r in subset if r['phase'] in ('retract','coast')
                            and r['projected_gap_mm']>2.0),None)
            # The cube's front edge must first cross PATCH_BEGIN *after* separation.
            overlap=next((r['time_s'] for r in subset
                          if r['footprint_patch_overlap']),None)
            # Measured first time where geometry indicates the pusher is 2 mm clear.
            # This is a heuristic, not contact sensor confirmation.
            separation_time=(separated or {}).get('time_s')
            patch_after_separation=(separation_time is not None and
                                    overlap is not None and
                                    overlap > separation_time + DT)
            # Last time the cube had appreciable forward sliding velocity.
            moving=[r['time_s'] for r in subset if abs(r['vx_m_s']) > 0.005]
            stop_time=max(moving) if moving else None
            start_z=subset[0]['cube_z_m']
            entry=dict(hz=args.hz,lane=name,
                       end_push_dx_mm=push_end['displacement_mm'],
                       end_push_vx_m_s=push_end['vx_m_s'],
                       first_separated_time_s=(separated or {}).get('time_s'),
                       first_separated_vx_m_s=(separated or {}).get('vx_m_s'),
                       final_dx_mm=final['displacement_mm'],
                       post_push_dx_mm=final['displacement_mm']-push_end['displacement_mm'],
                       peak_abs_pitch_deg=max(abs(r['pitch_deg']) for r in subset),
                       max_z_rise_mm=1000*max(r['cube_z_m']-start_z for r in subset),
                       first_patch_overlap_time_s=overlap,
                       min_gap_mm=min(r['projected_gap_mm'] for r in subset),
                       separation_dx_mm=(separated or {}).get('displacement_mm'),
                       post_separation_dx_mm=(final['displacement_mm']-separated['displacement_mm'])
                           if separated is not None else None,
                       patch_after_separation=patch_after_separation,
                       separation_to_patch_s=(overlap-separation_time)
                           if overlap is not None and separation_time is not None else None,
                       stop_time_s=stop_time)
            w.writerow(entry)
            print(f'{args.hz} Hz {name:8s} push_vx={entry["end_push_vx_m_s"]:+.3f} m/s '
                  f'sep_vx={entry["first_separated_vx_m_s"]} '
                  f'final_dx={entry["final_dx_mm"]:.3f} mm '
                  f'post_push={entry["post_push_dx_mm"]:+.3f} mm '
                  f'pitch={entry["peak_abs_pitch_deg"]:.3f} deg '
                  f'separation_t={separation_time} '
                  f'patch_t={overlap} '
                  f'patch_after_sep={patch_after_separation} '
                  f'coast_after_sep={entry["post_separation_dx_mm"]}',flush=True)
    print('Saved:',traj,'\nSaved:',summary)
    print(f'Patch begins at x={PATCH_BEGIN*1000:.1f} mm, '
          f'cube front initially at x={CUBE_SIZE*500:.1f} mm.')
    print('PASS criterion: patch_after_sep=True for patch lane at each frequency.')
    print('CAUTION: patch encounter and separation are inferred geometrically; no contact impulses measured.')
    print('No cube velocity was assigned. The pusher is pose-stepped and kinematic.')
    if not args.headless:
        print('Close Isaac Sim window to exit.')
        while app.is_running():
            app.update()
except Exception:
    traceback.print_exc()
    raise
finally:
    app.close()
