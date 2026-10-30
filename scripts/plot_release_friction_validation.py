"""Plot Module 1A.3c release friction experiments; regular Python environment.
Run: python scripts/plot_release_friction_validation.py
"""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
frames = []
for hz in (120, 240):
    p = OUT / f'release_friction_{hz}hz.csv'
    if not p.exists():
        raise SystemExit(f'Missing {p}. Run both Isaac Sim frequencies first.')
    frames.append(pd.read_csv(p))
data = pd.concat(frames, ignore_index=True)

for y, ylabel, filename in [
    ('dx_mm', 'Object X displacement (mm)', 'release_friction_displacement.png'),
    ('vx_m_s', 'Object X velocity (m/s)', 'release_friction_velocity.png'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    for ax, condition in zip(axes, ('nominal', 'localized_patch')):
        d = data[data.condition == condition]
        for hz, g in d.groupby('physics_hz'):
            ax.plot(g.time_s, g[y], label=f'{hz} Hz', linewidth=1.8)
        ax.set(title=condition, xlabel='Time after velocity release (s)')
        ax.grid(alpha=0.25)
        ax.legend()
    axes[0].set_ylabel(ylabel)
    fig.suptitle('Module 1A.3c | Controlled initial-velocity friction test')
    fig.tight_layout()
    path = OUT / filename
    fig.savefig(path, dpi=180)
    print('Saved:', path)
    plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, hz in zip(axes, (120, 240)):
    d = data[data.physics_hz == hz]
    for condition, g in d.groupby('condition'):
        ax.plot(g.time_s, g.dx_mm, label=condition, linewidth=1.8)
    ax.set(title=f'{hz} Hz', xlabel='Time after velocity release (s)')
    ax.grid(alpha=0.25)
    ax.legend()
axes[0].set_ylabel('Object X displacement (mm)')
fig.suptitle('Module 1A.3c | Nominal vs localized patch')
fig.tight_layout()
path = OUT / 'release_friction_patch_effect.png'
fig.savefig(path, dpi=180)
print('Saved:', path)
plt.close(fig)

summaries = pd.concat([pd.read_csv(OUT / f'release_friction_summary_{hz}hz.csv')
                       for hz in (120, 240)], ignore_index=True)
print('\nSummary:\n', summaries.to_string(index=False))
