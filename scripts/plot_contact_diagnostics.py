"""Compare Module 1A.3 contact diagnostics at 120 Hz versus 240 Hz.
Run with normal Python: python scripts/plot_contact_diagnostics.py
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / 'results'
frames = []
for hz in (120, 240):
    path = RESULTS / f'contact_diagnostics_{hz}hz.csv'
    if not path.exists():
        raise SystemExit(f'Missing {path.name}: run the --hz {hz} simulation first.')
    frame = pd.read_csv(path)
    expected = {'time_s', 'condition', 'dx_mm', 'vx_m_s', 'projected_gap_mm',
                'phase', 'pusher_command_x_m', 'pusher_actual_x_m'}
    if not expected.issubset(frame.columns):
        raise SystemExit(f'Missing columns in {path.name}: {expected - set(frame.columns)}')
    frame['hz'] = hz
    frames.append(frame)
df = pd.concat(frames, ignore_index=True)

for col, ylabel, filename in [
    ('dx_mm', 'Cube X displacement (mm)', 'contact_displacement.png'),
    ('vx_m_s', 'Cube X velocity (m/s)', 'contact_velocity.png'),
    ('projected_gap_mm', 'Projected X clearance (mm)', 'contact_gap.png'),
]:
    fig, ax = plt.subplots(figsize=(11, 5))
    for (hz, condition), sub in df.groupby(['hz', 'condition'], sort=True):
        ax.plot(sub.time_s, sub[col], label=f'{condition} — {hz} Hz',
                linewidth=1.2, alpha=.85, linestyle='-' if hz == 120 else '--')
    ax.axvline(.4, ls=':', color='gray', label='push ends')
    ax.axvline(.8, ls='-.', color='gray', label='retract ends')
    ax.set(title=f'Module 1A.3 | {ylabel}', xlabel='Time (s)', ylabel=ylabel)
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS / filename, dpi=170)
    plt.close(fig)

# Each axis shows a different physical condition; timestep changes are overlaid.
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, condition in zip(axes, ['nominal', 'localized_patch']):
    for hz in (120, 240):
        sub = df[(df.condition == condition) & (df.hz == hz)]
        ax.plot(sub.time_s, sub.vx_m_s, label=f'{hz} Hz',
                linestyle='-' if hz == 120 else '--')
    ax.set(title=condition, xlabel='Time (s)', xlim=(.17, .5))
    ax.grid(alpha=.25)
    ax.legend()
axes[0].set_ylabel('Cube X velocity (m/s)')
fig.suptitle('Contact velocity oscillations — timestep comparison')
fig.tight_layout()
fig.savefig(RESULTS / 'contact_velocity_zoom.png', dpi=170)
plt.close(fig)

summary = []
for hz in (120, 240):
    summary.append(pd.read_csv(RESULTS / f'contact_diagnostics_summary_{hz}hz.csv'))
summary = pd.concat(summary, ignore_index=True)
print('\nDiagnostics summary:')
print(summary.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
for condition in ('nominal', 'localized_patch'):
    sub = summary[summary.condition == condition].set_index('physics_hz')
    d = sub.loc[240, 'dx_final_mm'] - sub.loc[120, 'dx_final_mm']
    print(f'{condition}: final displacement difference (240 - 120 Hz) = {d:+.4f} mm')
print('\nSaved displacement, velocity, gap, and velocity-zoom plots to:', RESULTS)
