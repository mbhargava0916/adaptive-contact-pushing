"""Plot Module 1A.3c localized-patch results. Run with system Python."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parents[1] / 'results'
frames = []
for hz in (120, 240):
    path = OUT / f'localized_patch_clockfix_{hz}hz.csv'
    if not path.exists():
        raise SystemExit(f'Missing {path}; run both simulation frequencies first')
    df = pd.read_csv(path)
    df['frequency'] = hz
    frames.append(df)
data = pd.concat(frames, ignore_index=True)

for y_col, y_label, filename in [
    ('dx_mm', 'X displacement (mm)', 'localized_patch_displacement.png'),
    ('vx_m_s', 'X velocity (m/s)', 'localized_patch_velocity.png'),
]:
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for ax, hz in zip(axs, (120, 240)):
        for label in ('nominal', 'localized_patch'):
            d = data[(data.frequency == hz) & (data.condition == label)]
            ax.plot(d.time_s, d[y_col], label=label)
        ax.set_title(f'{hz} Hz')
        ax.set_xlabel('Time after velocity release (s)')
        ax.grid(alpha=0.3)
        ax.legend()
    axs[0].set_ylabel(y_label)
    fig.suptitle('Module 1A.3c | Nominal vs localized friction patch')
    fig.tight_layout()
    fig.savefig(OUT / filename, dpi=180)
    plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
for ax, hz in zip(axs, (120, 240)):
    d = data[data.frequency == hz].pivot(index='time_s', columns='condition', values='dx_mm')
    difference = d['nominal'] - d['localized_patch']
    ax.plot(difference.index, difference.values)
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_title(f'{hz} Hz')
    ax.set_xlabel('Time after release (s)')
    ax.grid(alpha=0.3)
axs[0].set_ylabel('Nominal − patch displacement (mm)')
fig.suptitle('Module 1A.3c | Difference attributable to patch lane')
fig.tight_layout()
fig.savefig(OUT / 'localized_patch_difference.png', dpi=180)
plt.close(fig)
print('Saved displacement, velocity, and difference plots to:', OUT)

for hz in (120, 240):
    summary = pd.read_csv(OUT / f'localized_patch_clockfix_summary_{hz}hz.csv')
    vals = dict(zip(summary.condition, summary.final_dx_mm))
    print(f'{hz} Hz: nominal={vals["nominal"]:.3f} mm, '
          f'patch={vals["localized_patch"]:.3f} mm, '
          f'difference={vals["nominal"]-vals["localized_patch"]:.3f} mm')
