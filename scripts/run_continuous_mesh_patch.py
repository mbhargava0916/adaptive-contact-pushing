"""Module 1A.3c: continuous box vs single-mesh uniform vs single-mesh patch.

Run via Isaac Sim 5.1 Python:
  C:\\isaacsim\\python.bat scripts\\run_continuous_mesh_patch.py --hz 240
  C:\\isaacsim\\python.bat scripts\\run_continuous_mesh_patch.py --hz 120

This is a *diagnostic*, not a validated friction model. In particular, USD
GeomSubset physics-material interpretation must be verified in your runtime.
"""
import argparse
import csv
import math
import os
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--hz', type=int, default=240, choices=(120, 240, 480))
p.add_argument('--headless', action='store_true')
p.add_argument('--speed', type=float, default=0.7)
a = p.parse_args()
if a.speed <= 0:
    p.error('--speed must be positive')
os.environ.setdefault('OMNI_KIT_ACCEPT_EULA', 'YES')
from isaacsim import SimulationApp
app = SimulationApp({'headless': a.headless, 'renderer': 'RaytracedLighting'})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid
from isaacsim.core.api.materials import PhysicsMaterial
from isaacsim.core.utils.viewports import set_camera_view
from pxr import Gf, Vt, UsdGeom, UsdPhysics, UsdShade, UsdLux, Sdf

DT = 1.0 / a.hz
CUBE_SIZE = 0.05
THICKNESS = 0.05
X_POINTS = [-0.35, 0.05, 0.15, 0.65]
HALF_WIDTH = 0.09
Y_LANES = {'continuous_box': -0.25, 'uniform_mesh': 0.0, 'patch_mesh': 0.25}
OUT = Path(__file__).resolve().parents[1] / 'results'
OUT.mkdir(parents=True, exist_ok=True)


def pitch_degrees(quat):
    w, x, y, z = map(float, quat)
    return math.degrees(math.asin(max(-1., min(1., 2 * (w*y - z*x)))))


def build_mesh(stage, path, y, nominal_mat_path, patch_mat_path, enable_patch):
    """One closed, watertight static mesh with a subdivided, perfectly planar top.

    Top triangles share EXACT vertex indices along x=0.05 and x=0.15.
    There are no coincident overlapping boxes or raised patch geometry.
    """
    mesh = UsdGeom.Mesh.Define(stage, path)
    points = []
    for z in (0.0, -THICKNESS):
        for x in X_POINTS:
            for side in (-HALF_WIDTH, HALF_WIDTH):
                points.append(Gf.Vec3f(x, y + side, z))
    # index: zlevel * 8 + 2*x_index + y_index
    def idx(z, j, side):
        return 8 * z + 2 * j + side
    tris, top_patch_faces = [], []
    def add_tri(i, j, k, patch=False):
        face_id = len(tris)
        tris.append((i, j, k))
        if patch:
            top_patch_faces.append(face_id)
    for seg in range(3):
        # Top, upward-normal facing. The middle segment is the patch.
        v00, v01 = idx(0, seg, 0), idx(0, seg, 1)
        v10, v11 = idx(0, seg + 1, 0), idx(0, seg + 1, 1)
        add_tri(v00, v10, v11, seg == 1)
        add_tri(v00, v11, v01, seg == 1)
        # Bottom, downward facing
        b00, b01 = idx(1, seg, 0), idx(1, seg, 1)
        b10, b11 = idx(1, seg + 1, 0), idx(1, seg + 1, 1)
        add_tri(b00, b11, b10)
        add_tri(b00, b01, b11)
        # side at y=-HALF_WIDTH
        add_tri(v00, b00, b10)
        add_tri(v00, b10, v10)
        # side at y=+HALF_WIDTH
        add_tri(v01, v11, b11)
        add_tri(v01, b11, b01)
    # close x ends
    for seg in (0, 3):
        va, vb = idx(0, seg, 0), idx(0, seg, 1)
        ba, bb = idx(1, seg, 0), idx(1, seg, 1)
        if seg == 0:
            add_tri(va, vb, bb)
            add_tri(va, bb, ba)
        else:
            add_tri(va, bb, vb)
            add_tri(va, ba, bb)
    mesh.CreatePointsAttr(Vt.Vec3fArray(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([v for t in tris for v in t]))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(False)
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    # USD Physics, not PhysxSchema: force a static triangle-mesh collider.
    # Approximation "none" preserves the authored triangles rather than a convex hull.
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
    mesh_collision.CreateApproximationAttr().Set('none')
    print(f'{path}: collision approximation = {mesh_collision.GetApproximationAttr().Get()}')
    binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
    nominal_mat = UsdShade.Material.Get(stage, nominal_mat_path)
    patch_mat = UsdShade.Material.Get(stage, patch_mat_path)
    assert nominal_mat and patch_mat, 'PhysicsMaterial should create USD Material prims'
    binding.Bind(nominal_mat, UsdShade.Tokens.weakerThanDescendants,
                 'physics')
    if enable_patch:
        subset = binding.CreateMaterialBindSubset('HighFrictionTopFaces', Vt.IntArray(top_patch_faces))
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
            patch_mat, UsdShade.Tokens.weakerThanDescendants, 'physics')
        print(f'{path}: authored high-friction subset on top triangles {top_patch_faces}')
    return mesh


try:
    world = World(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
    nominal = PhysicsMaterial('/World/Materials/Nominal', static_friction=0.75,
                              dynamic_friction=0.65, restitution=0.0)
    high = PhysicsMaterial('/World/Materials/High', static_friction=0.75,
                           dynamic_friction=0.65, restitution=0.0)
    cube_mat = PhysicsMaterial('/World/Materials/Cube', static_friction=0.3,
                               dynamic_friction=0.25, restitution=0.0)
    cubes = {}
    for name, y in Y_LANES.items():
        if name == 'continuous_box':
            table = world.scene.add(FixedCuboid(
                prim_path='/World/continuous_box/Table', name='continuous_box_table',
                position=np.array([0.15, y, -THICKNESS/2]),
                size=1.0, scale=np.array([1.0, 2*HALF_WIDTH, THICKNESS]),
                color=np.array([0.64, 0.67, 0.72])))
            table.apply_physics_material(nominal)
        else:
            build_mesh(world.stage, f'/World/{name}/Table', y,
                       '/World/Materials/Nominal', '/World/Materials/High',
                       name == 'patch_mesh')
        cube = world.scene.add(DynamicCuboid(
            prim_path=f'/World/{name}/Cube', name=f'{name}_cube',
            position=np.array([0., y, 0.026]),
            size=CUBE_SIZE, mass=0.2, color=np.array([0.85, 0.24, 0.17])))
        cube.apply_physics_material(cube_mat)
        cubes[name] = cube
    light = UsdLux.DistantLight.Define(world.stage, '/World/KeyLight')
    light.CreateIntensityAttr(2000.)
    if not a.headless:
        set_camera_view(eye=np.array([0.46, -1.1, 0.8]),
                        target=np.array([0.1, 0., 0.]))

    world.reset()
    for _ in range(round(0.7/DT)):
        world.step(render=not a.headless)
    starts = {}
    for name, cube in cubes.items():
        pos, _ = cube.get_world_pose()
        starts[name] = np.array(pos, dtype=float)
        cube.set_angular_velocity(np.zeros(3))
        cube.set_linear_velocity(np.array([a.speed, 0., 0.]))

    t0 = float(world.current_time)
    records = []
    def record():
        t = float(world.current_time) - t0
        for name, cube in cubes.items():
            pos, q = cube.get_world_pose()
            pos = np.array(pos, dtype=float)
            vx, vy, vz = map(float, cube.get_linear_velocity())
            wx, wy, wz = map(float, cube.get_angular_velocity())
            records.append(dict(hz=a.hz, condition=name, time_s=t,
                                dx_mm=(pos[0]-starts[name][0])*1000,
                                z_rise_mm=(pos[2]-starts[name][2])*1000,
                                x_m=pos[0], z_m=pos[2], vx_m_s=vx, vz_m_s=vz,
                                pitch_deg=pitch_degrees(q), omega_y_rad_s=wy,
                                footprint_patch_overlap=int(pos[0] + CUBE_SIZE/2 > 0.05 and
                                                            pos[0] - CUBE_SIZE/2 < 0.15)))
    record()
    for _ in range(round(2./DT)):
        world.step(render=not a.headless)
        record()
    print('World clock elapsed: {:.6f} s (expected 2.000000 s)'.format(float(world.current_time)-t0))

    traj = OUT / f'continuous_mesh_patch_{a.hz}hz.csv'
    with traj.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = OUT / f'continuous_mesh_patch_summary_{a.hz}hz.csv'
    cols = ['hz','condition','final_dx_mm','max_abs_pitch_deg',
            'max_rise_mm','min_vx_m_s','first_footprint_overlap_s']
    sums = {}
    with summary.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for name in Y_LANES:
            rows = [r for r in records if r['condition'] == name]
            overlap = next((r['time_s'] for r in rows if r['footprint_patch_overlap']), None)
            row = dict(hz=a.hz, condition=name, final_dx_mm=rows[-1]['dx_mm'],
                       max_abs_pitch_deg=max(abs(r['pitch_deg']) for r in rows),
                       max_rise_mm=max(r['z_rise_mm'] for r in rows),
                       min_vx_m_s=min(r['vx_m_s'] for r in rows),
                       first_footprint_overlap_s=overlap)
            writer.writerow(row)
            sums[name] = row
            print(f'{a.hz:3d} Hz {name:16s} dx={row["final_dx_mm"]:8.3f} mm '
                  f'pitch={row["max_abs_pitch_deg"]:6.3f} deg '
                  f'rise={row["max_rise_mm"]:7.3f} mm '
                  f'min_vx={row["min_vx_m_s"]:+.3f} m/s')
    print('Mesh uniform - box: {:.3f} mm'.format(sums['uniform_mesh']['final_dx_mm'] - sums['continuous_box']['final_dx_mm']))
    print('Mesh patch - uniform mesh: {:.3f} mm'.format(sums['patch_mesh']['final_dx_mm'] - sums['uniform_mesh']['final_dx_mm']))
    print('Saved:', traj, '\nSaved:', summary)
    print('CAUTION: face-subset binding is authored, but physical enforcement must be verified experimentally.')
    if not a.headless:
        print('Close Isaac Sim window to exit.')
        while app.is_running():
            app.update()
except Exception:
    import traceback
    print('ERROR during continuous-mesh validation:', flush=True)
    traceback.print_exc()
    raise
finally:
    app.close()
