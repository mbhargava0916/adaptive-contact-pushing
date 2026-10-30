"""Plot Module 1A.3 velocity and displacement, with stage boundaries."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
results = root / 'results'
df = pd.read_csv(results / 'friction_patch_trajectories.csv')
colors = {'low': 'tab:blue', 'nominal': 'tab:orange',
          'high_uniform': 'tab:green', 'localized_patch': 'tab:red'}
for ykey, ylabel, filename in [
    ('dx_mm', 'Cube X displacement (mm)', 'friction_patch_displacement.png'),
    ('vx_m_s', 'Cube X velocity (m/s)', 'friction_patch_velocity.png')]:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for label, group in df.groupby('condition', sort=False):
        group = group.sort_values('time_s')
        ax.plot(group['time_s'], group[ykey], label=label.replace('_', ' '),
                color=colors.get(label), linewidth=1.8)
    ax.axvline(0.4, color='black', ls='--', alpha=.6, label='push ends')
    ax.axvline(0.8, color='gray', ls=':', alpha=.8, label='retract ends')
    ax.set(xlabel='Time since push begins (s)', ylabel=ylabel,
           title=f'Module 1A.3 | {ylabel} — uniform vs local patch')
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    dest = results / filename
    fig.savefig(dest, dpi=180)
    print('Saved:', dest)
    plt.close(fig)

print('\nFinal displacement (mm):')
print(pd.read_csv(results / 'friction_patch_summary.csv')[['condition', 'dx_mm']].to_string(index=False))
