"""Compare Module 1A.3b 120 Hz and 240 Hz results (optional 480 Hz)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results'
frames = []
for hz in (120, 240, 480):
    path = OUT / f'smooth_contact_{hz}hz.csv'
    if path.exists():
        frames.append(pd.read_csv(path))
if not frames:
    raise SystemExit('No smooth_contact_*hz.csv files found. Run Isaac Sim first.')
all_data = pd.concat(frames, ignore_index=True)
hz_set = sorted(all_data.physics_hz.unique())
conditions = ('nominal', 'localized_patch')

for metric, ylabel, filename in [
    ('dx_mm', 'Object X displacement (mm)', 'smooth_contact_displacement.png'),
    ('vx_m_s', 'Object X velocity (m/s)', 'smooth_contact_velocity.png'),
    ('projected_gap_mm', 'Projected pusher-to-cube gap (mm)', 'smooth_contact_gap.png'),
]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
    for ax, condition in zip(axes, conditions):
        for hz in hz_set:
            d = all_data[(all_data.condition == condition) &
                         (all_data.physics_hz == hz)]
            ax.plot(d.time_s, d[metric], label=f'{hz} Hz', linewidth=1.2)
        ax.axvline(.65, linestyle='--', color='gray', linewidth=.8, label='push ends')
        ax.axvline(1.10, linestyle=':', color='gray', linewidth=.8, label='retract ends')
        ax.set_title(condition.replace('_', ' '))
        ax.set_xlabel('Time since push begins (s)')
        ax.grid(alpha=.25)
    axes[0].set_ylabel(ylabel)
    axes[0].legend()
    fig.suptitle('Module 1A.3b | Smooth push timestep comparison')
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)
    print('Saved:', OUT / filename)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)
for ax, condition in zip(axes, conditions):
    for hz in hz_set:
        d = all_data[(all_data.condition == condition) &
                     (all_data.physics_hz == hz) &
                     (all_data.time_s.between(.20, .70))]
        ax.plot(d.time_s, d.vx_m_s, label=f'{hz} Hz', linewidth=1.1)
    ax.set_title(condition.replace('_', ' '))
    ax.set_xlabel('Time (s)')
    ax.grid(alpha=.25)
axes[0].set_ylabel('Object X velocity (m/s)')
axes[0].legend()
fig.suptitle('Module 1A.3b | Contact velocity (zoom)')
fig.tight_layout()
fig.savefig(OUT / 'smooth_contact_velocity_zoom.png', dpi=180)
plt.close(fig)
print('Saved:', OUT / 'smooth_contact_velocity_zoom.png')

summaries = []
for hz in hz_set:
    p = OUT / f'smooth_contact_summary_{hz}hz.csv'
    if p.exists():
        summaries.append(pd.read_csv(p))
if summaries:
    summary = pd.concat(summaries, ignore_index=True)
    print('\nFinal displacements (mm):')
    print(summary.pivot(index='physics_hz', columns='condition',
                        values='final_dx_mm').round(4).to_string())
