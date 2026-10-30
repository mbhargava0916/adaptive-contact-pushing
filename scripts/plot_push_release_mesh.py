"""Plot Module 1A.3d two-lane pushing diagnostics.
Run: python scripts/plot_push_release_mesh.py
Requires: pandas matplotlib
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
files = [(hz, OUT/f'push_release_mesh_refined_{hz}hz.csv') for hz in (120,240,480)]
found = [(hz,p) for hz,p in files if p.exists()]
if not found:
    raise SystemExit('No CSVs found; run Isaac Sim simulation first.')
for metric,units,name in [
    ('displacement_mm','mm','displacement'),
    ('vx_m_s','m/s','velocity'),
    ('pitch_deg','degrees','pitch')]:
    fig, ax = plt.subplots(figsize=(10,5.5))
    for hz,path in found:
        df=pd.read_csv(path)
        for lane,g in df.groupby('lane'):
            ax.plot(g['time_s'],g[metric],label=f'{lane} ({hz} Hz)',
                    linewidth=1.5,alpha=0.85)
    ax.set(xlabel='Time since settled start (s)',ylabel=f'{metric} ({units})',
           title=f'Contact push and release: {name}')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    target=OUT/f'push_release_mesh_refined_{name}.png'
    fig.savefig(target,dpi=180)
    print('Saved:',target)
    plt.close(fig)

fig,ax=plt.subplots(figsize=(10,5.5))
for hz,path in found:
    df=pd.read_csv(path)
    for lane,g in df.groupby('lane'):
        ax.plot(g.time_s,g.projected_gap_mm,label=f'{lane} ({hz} Hz)',alpha=0.8)
ax.axhline(0,color='gray',linestyle='--',linewidth=1)
ax.set(xlabel='Time (s)',ylabel='Approximate projected gap (mm)',
       title='Pusher-to-cube clearance (geometric approximation)')
ax.grid(alpha=0.25);ax.legend();fig.tight_layout()
p=OUT/'push_release_mesh_refined_gap.png';fig.savefig(p,dpi=180)
print('Saved:',p)
plt.close(fig)

# Event timing: patch footprint encounter compared to separation estimate.
for hz, path in found:
    summary = OUT / f'push_release_mesh_refined_summary_{hz}hz.csv'
    if summary.exists():
        print(f"\n{hz} Hz event diagnostics:")
        d = pd.read_csv(summary)
        print(d[['lane','first_separated_time_s','first_patch_overlap_time_s',
                 'patch_after_separation','post_separation_dx_mm','final_dx_mm']].to_string(index=False))
