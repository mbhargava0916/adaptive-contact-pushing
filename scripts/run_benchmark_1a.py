"""Module 1A.4 — memory-conscious, crash-resumable 30-trial Isaac Sim benchmark.

Cylinder workaround (experimental): /physics/collisionApproximateCylinders=True
for cylinder trials to avoid native specialized cylinder/triangle-mesh contact path.

Run from repository root (Windows):
  C:\\Isaac\\isaac-sim-standalone-5.1.0-windows-x86_64\\python.bat scripts\\run_benchmark_1a.py --headless --smoke
  C:\\Isaac\\isaac-sim-standalone-5.1.0-windows-x86_64\\python.bat scripts\\run_benchmark_1a.py --headless

A fresh Isaac Sim process runs ONE physical scene and ONE trial, then exits.
Outputs: results/benchmark_1a/{manifest.json,trials.csv,trajectories.csv,individual/}
The controller never assigns cube velocities; it drives a kinematic pusher.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "benchmark_1a"
PARTS = OUT / "individual"
SHAPES = ("cube", "rectangle", "cylinder")
CONTACTS = ("nominal", "low", "patch")
TARGET_MM = 89.0
POSITION_TOL_MM = 20.0
YAW_TOL_DEG = 5.0


def arguments():
    p = argparse.ArgumentParser(description="Sequential Isaac Sim Module 1A.4 benchmark")
    p.add_argument("--hz", type=int, choices=(120, 240, 480), default=240)
    p.add_argument("--seed", type=int, default=20261030)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Run only T01 to verify scene")
    p.add_argument("--trial", type=str, help="Run only trial ID, e.g. T14")
    p.add_argument("--no-resume", action="store_true", help="Rerun even completed trials")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--config", type=str, help=argparse.SUPPRESS)
    return p.parse_args()


def generate_configs(seed):
    import random
    rng = random.Random(seed)
    plans = []
    for pass_num in range(3):
        for shape in SHAPES:
            for contact in CONTACTS:
                reference = pass_num == 0
                plans.append(dict(
                    trial_id=f"T{len(plans)+1:02d}", shape=shape, contact=contact,
                    pass_num=pass_num, seed=seed, is_repeat=False, repeat_of="",
                    offset_x_m=0.0 if reference else rng.uniform(-.003, .003),
                    offset_y_m=0.0 if reference else rng.uniform(-.003, .003),
                    initial_yaw_deg=0.0 if reference else rng.uniform(-3., 3.),
                    mass_kg=.2 if reference else rng.uniform(.18, .22)))
    for i, shape in enumerate(SHAPES):
        plans.append(dict(trial_id=f"T{len(plans)+1:02d}", shape=shape,
            contact="nominal", pass_num=3, seed=seed, is_repeat=True,
            repeat_of=f"T{1+3*i:02d}", offset_x_m=0., offset_y_m=0.,
            initial_yaw_deg=0., mass_kg=.2))
    assert len(plans) == 30
    return plans


def write_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate(plans):
    summaries = []
    trajectories = []
    for config in plans:
        name = config["trial_id"]
        a, b = PARTS / f"{name}_summary.json", PARTS / f"{name}_trajectory.csv"
        if a.is_file() and b.is_file():
            summaries.append(json.loads(a.read_text(encoding="utf-8")))
            trajectories.extend(read_csv(b))
    write_csv(OUT / "trials.csv", summaries)
    write_csv(OUT / "trajectories.csv", trajectories)
    return summaries


def coordinator(args):
    OUT.mkdir(parents=True, exist_ok=True)
    PARTS.mkdir(parents=True, exist_ok=True)
    plans = generate_configs(args.seed)
    # Avoid mixing partial outputs from different seeds or timesteps.
    manifest_file = OUT / "manifest.json"
    manifest = dict(seed=args.seed, hz=args.hz, target_dx_mm=TARGET_MM,
        position_tolerance_mm=POSITION_TOL_MM, yaw_tolerance_deg=YAW_TOL_DEG,
        configurations=plans, execution="one fresh Isaac Sim process per trial")
    if manifest_file.exists() and not args.no_resume:
        prior = json.loads(manifest_file.read_text(encoding="utf-8"))
        if prior.get("seed") != args.seed or prior.get("hz") != args.hz:
            raise SystemExit("Existing output has different --seed or --hz. Move results/benchmark_1a elsewhere, or use matching options.")
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    selected = [p for p in plans if (not args.smoke or p["trial_id"] == "T01")
                                  and (not args.trial or p["trial_id"] == args.trial.upper())]
    if not selected:
        raise SystemExit("No matching trials")
    python = sys.executable
    print(f"[Benchmark] Executable: {python}", flush=True)
    print(f"[Benchmark] {len(selected)} trial(s), each in a new Isaac Sim process", flush=True)
    for i, config in enumerate(selected, start=1):
        name = config["trial_id"]
        summary_file = PARTS / f"{name}_summary.json"
        trace_file = PARTS / f"{name}_trajectory.csv"
        if not args.no_resume and summary_file.exists() and trace_file.exists():
            print(f"[{i}/{len(selected)}] {name} already complete; skipping", flush=True)
            continue
        # Avoid accidentally accepting a stale summary after a crash.
        summary_file.unlink(missing_ok=True)
        trace_file.unlink(missing_ok=True)
        command = [python, str(Path(__file__).resolve()), "--worker",
                   "--config", json.dumps(config), "--hz", str(args.hz)]
        if args.headless:
            command.append("--headless")
        print(f"[{i}/{len(selected)}] Running {name}: {config['shape']}/{config['contact']}", flush=True)
        completed = subprocess.run(command, cwd=str(ROOT), check=False)
        if completed.returncode != 0 or not summary_file.exists() or not trace_file.exists():
            aggregate(plans)
            print(f"[Benchmark] FAILED on {name}, return code {completed.returncode}. "
                  "Completed trials preserved; fix issue and rerun to resume.", flush=True)
            raise SystemExit(completed.returncode or 1)
        aggregate(plans)
    rows = aggregate(plans)
    print(f"[Benchmark] Saved {len(rows)}/30 summaries to {OUT}", flush=True)
    if len(rows) == 30:
        print(f"[Benchmark] Success: {sum(int(r['success']) for r in rows)}/30", flush=True)
    elif args.smoke:
        print("[Benchmark] Smoke test complete. Run without --smoke for full benchmark.", flush=True)


def worker(args):
    # Import Isaac Sim ONLY in the child. Parent never loads simulation libraries.
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": args.headless, "renderer": "RaytracedLighting"})
    try:
        import numpy as np
        from isaacsim.core.api import World
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
        from isaacsim.core.api.materials import PhysicsMaterial
        from pxr import Gf, Vt, UsdGeom, UsdPhysics, UsdShade, UsdLux
        config = json.loads(args.config)
        hz = args.hz
        dt = 1.0 / hz
        patch_start, patch_end = .105, .205
        xpoints = [-.35, patch_start, patch_end, .65]
        half_width = .09
        pusher_start = -.09
        pusher_z = .025
        push_duration = .18
        ramp_duration = .035
        travel = .095
        retract_duration = .14
        coast_duration = 1.5
        settle_duration = .65

        def euler_yaw(q):
            w, x, y, z = map(float, q)
            return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

        def euler_pitch(q):
            w, x, y, z = map(float, q)
            return math.degrees(math.asin(max(-1., min(1., 2*(w*y-z*x)))))

        def delta_angle(a, b):
            return (a-b+180) % 360-180

        def push_x(t):
            speed = travel / (push_duration - .5*ramp_duration)
            if t <= ramp_duration:
                return pusher_start + .5*speed*t*t/ramp_duration
            return pusher_start + speed*(t-.5*ramp_duration)

        world = World(physics_dt=dt, rendering_dt=dt, stage_units_in_meters=1.)
        surface_material = {}
        for label, sta, dyn in (("nominal", .45, .35), ("low", .05, .04),
                                 ("high", 1.2, 1.), ("object", .45, .35),
                                 ("pusher", .5, .4)):
            surface_material[label] = PhysicsMaterial(
                f"/World/Materials/{label}", static_friction=sta,
                dynamic_friction=dyn, restitution=0.)

        # One watertight/static triangle-mesh table; only top-face friction varies.
        mesh = UsdGeom.Mesh.Define(world.stage, "/World/Table")
        pts = [Gf.Vec3f(x, side*half_width, z)
               for z in (0., -.05) for x in xpoints for side in (-1, 1)]
        def ix(layer, j, side):
            return layer*8+2*j+side
        tris, patch_faces = [], []
        def tri(a, b, c, is_patch=False):
            if is_patch:
                patch_faces.append(len(tris))
            tris.append((a, b, c))
        for j in range(3):
            a, b, c, d = ix(0,j,0), ix(0,j,1), ix(0,j+1,0), ix(0,j+1,1)
            e, f, g, h = ix(1,j,0), ix(1,j,1), ix(1,j+1,0), ix(1,j+1,1)
            tri(a,c,d,j==1);tri(a,d,b,j==1)
            tri(e,h,g);tri(e,f,h)
            tri(a,e,g);tri(a,g,c)
            tri(b,d,h);tri(b,h,f)
        for j in (0,3):
            a,b,e,f = ix(0,j,0),ix(0,j,1),ix(1,j,0),ix(1,j,1)
            if j == 0: tri(a,b,f);tri(a,f,e)
            else: tri(a,f,b);tri(a,e,f)
        mesh.CreatePointsAttr(Vt.Vec3fArray(pts))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3]*len(tris)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray([v for tup in tris for v in tup]))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")
        binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        bg = "low" if config["contact"] == "low" else "nominal"
        binding.Bind(UsdShade.Material.Get(world.stage, f"/World/Materials/{bg}"),
                     UsdShade.Tokens.weakerThanDescendants, "physics")
        if config["contact"] == "patch":
            subset = binding.CreateMaterialBindSubset("HighFrictionTopFaces", Vt.IntArray(patch_faces))
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(
                UsdShade.Material.Get(world.stage, "/World/Materials/high"),
                UsdShade.Tokens.weakerThanDescendants, "physics")

        shape = config["shape"]
        xoff = float(config["offset_x_m"])
        yoff = float(config["offset_y_m"])
        theta = math.radians(float(config["initial_yaw_deg"])) / 2
        q0 = np.array([math.cos(theta), 0., 0., math.sin(theta)])
        mass = float(config["mass_kg"])
        if shape == "cube":
            obj = DynamicCuboid(prim_path="/World/Object", name="trial_object",
                position=np.array([xoff,yoff,.026]), orientation=q0,
                size=.05,mass=mass,color=np.array([.90,.25,.13]))
        elif shape == "rectangle":
            obj = DynamicCuboid(prim_path="/World/Object", name="trial_object",
                position=np.array([xoff,yoff,.026]), orientation=q0,
                size=1.,scale=np.array([.07,.04,.05]),mass=mass,
                color=np.array([.85,.55,.14]))
        elif shape == "cylinder":
            obj = DynamicCylinder(prim_path="/World/Object", name="trial_object",
                position=np.array([xoff,yoff,.026]), orientation=q0,
                radius=.025,height=.05,mass=mass,color=np.array([.7,.22,.75]))
        else:
            raise ValueError(shape)
        obj = world.scene.add(obj)
        obj.apply_physics_material(surface_material["object"])
        pusher = world.scene.add(FixedCuboid(
            prim_path="/World/Pusher",name="trial_pusher",
            position=np.array([pusher_start,0.,pusher_z]),size=1.,
            scale=np.array([.02,.10,.05]),color=np.array([.15,.4,.95])))
        pusher.apply_physics_material(surface_material["pusher"])
        UsdPhysics.RigidBodyAPI.Apply(world.stage.GetPrimAtPath("/World/Pusher"))\
            .CreateKinematicEnabledAttr(True)
        light = UsdLux.DistantLight.Define(world.stage,"/World/KeyLight")
        light.CreateIntensityAttr(2000.)
        if not args.headless:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=np.array([.52,-.85,.56]), target=np.array([.07,0.,0.]))
        # Isaac Sim / PhysX can use specialized native cylinder-vs-triangle-mesh
        # contacts. T09 (cylinder + per-face-material triangle mesh) crashed at
        # world.reset() with a native access violation. Approximate cylinders as
        # convex shapes for this diagnostic, BEFORE physics initialization.
        if shape == "cylinder":
            import carb.settings
            settings = carb.settings.get_settings()
            settings.set("/physics/collisionApproximateCylinders", True)
            print(f"[Worker {config['trial_id']}] cylinder convex collision approximation enabled", flush=True)
        print(f"[Worker {config['trial_id']}] initializing physics ({shape}, {config['contact']})", flush=True)
        world.reset()
        for _ in range(round(settle_duration / dt)):
            world.step(render=not args.headless)
        p_start, q_start = obj.get_world_pose()
        p_start = np.asarray(p_start,dtype=float)
        yaw_start = euler_yaw(q_start)
        start_time = float(world.current_time)
        rows=[]
        phase="initial"
        def sample():
            pos, q = obj.get_world_pose()
            pos=np.asarray(pos,dtype=float)
            vel=np.asarray(obj.get_linear_velocity(),dtype=float)
            pushpos,_=pusher.get_world_pose()
            rows.append(dict(trial_id=config["trial_id"],shape=shape,contact=config["contact"],
                pass_num=config["pass_num"],time_s=float(world.current_time)-start_time,
                phase=phase,x_m=float(pos[0]),y_m=float(pos[1]),z_m=float(pos[2]),
                vx_m_s=float(vel[0]),displacement_x_mm=1000*float(pos[0]-p_start[0]),
                lateral_mm=1000*float(pos[1]-p_start[1]),
                yaw_change_deg=delta_angle(euler_yaw(q),yaw_start),
                pitch_deg=euler_pitch(q),pusher_x_m=float(pushpos[0])))
        sample()
        phase="push"
        n=round(push_duration/dt)
        for k in range(1,n+1):
            x=push_x(push_duration*k/n)
            pusher.set_world_pose(position=np.array([x,0.,pusher_z]))
            world.step(render=not args.headless)
            sample()
        phase="retract"
        end=push_x(push_duration)
        n=round(retract_duration/dt)
        for k in range(1,n+1):
            u=k/n
            smooth=3*u*u-2*u*u*u
            x=end+(pusher_start-end)*smooth
            pusher.set_world_pose(position=np.array([x,0.,pusher_z]))
            world.step(render=not args.headless)
            sample()
        phase="coast"
        for _ in range(round(coast_duration/dt)):
            world.step(render=not args.headless)
            sample()
        last=rows[-1]
        position_error=math.hypot(last["displacement_x_mm"]-TARGET_MM,last["lateral_mm"])
        yaw_error=abs(last["yaw_change_deg"])
        success=(position_error<=POSITION_TOL_MM and
                 (shape=="cylinder" or yaw_error<=YAW_TOL_DEG))
        summary=dict(**config,hz=hz,initial_x_m=float(p_start[0]),
            final_dx_mm=last["displacement_x_mm"],final_dy_mm=last["lateral_mm"],
            final_yaw_change_deg=last["yaw_change_deg"],position_error_mm=position_error,
            yaw_error_deg="" if shape=="cylinder" else yaw_error,
            peak_pitch_deg=max(abs(row["pitch_deg"]) for row in rows),
            success=int(success))
        PARTS.mkdir(parents=True, exist_ok=True)
        trace_file=PARTS/f"{config['trial_id']}_trajectory.csv"
        summary_file=PARTS/f"{config['trial_id']}_summary.json"
        write_csv(trace_file, rows)
        summary_file.write_text(json.dumps(summary,indent=2),encoding="utf-8")
        print(f"[Worker {config['trial_id']}] dx={summary['final_dx_mm']:.3f} mm, "
              f"dy={summary['final_dy_mm']:.3f} mm, error={position_error:.3f} mm, "
              f"success={success}", flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close()


if __name__ == "__main__":
    args = arguments()
    if args.worker:
        worker(args)
    else:
        coordinator(args)
