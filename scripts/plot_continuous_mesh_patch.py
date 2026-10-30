"""Plot Module 1A.3c continuous-mesh trials (requires pandas, matplotlib)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

folder = Path(__file__).resolve().parents[1] / 'results'
df = pd.concat([pd.read_csv(folder / f'continuous_mesh_patch_{hz}hz.csv') for hz in (120, 240)], ignore_index=True)
for metric, ylabel, suffix in (
    ('dx_mm', 'Displacement X (mm)', 'displacement'),
    ('pitch_deg', 'Pitch (degrees)', 'pitch'),
    ('z_rise_mm', 'Height change (mm)', 'rise'),
    ('vx_m_s', 'Velocity X (m/s)', 'velocity'),
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, hz in zip(axes, (120, 240)):
        for name in ('continuous_box', 'uniform_mesh', 'patch_mesh'):
            sub = df[(df.hz == hz) & (df.condition == name)]
            ax.plot(sub.time_s, sub[metric], label=name, linewidth=1.8)
        ax.set_title(f'{hz} Hz')
        ax.set_xlabel('Simulation time after release (s)')
        ax.grid(alpha=0.25)
    axes[0].set_ylabel(ylabel)
    axes[1].legend()
    fig.suptitle(f'Module 1A.3c | Continuous mesh validation — {suffix}')
    fig.tight_layout()
    out = folder / f'continuous_mesh_patch_{suffix}.png'
    fig.savefig(out, dpi=180)
    print('Saved:', out)
    plt.close(fig)
