"""Re-score Module 1A.4 against a fixed 0-degree final world-yaw target.

Run from repository root:
    python scripts/correct_benchmark_1a_metrics.py

Does NOT alter results/benchmark_1a/trials.csv or any simulation output.
"""
from pathlib import Path
import argparse
import pandas as pd
import matplotlib.pyplot as plt


def wrap_deg(angle):
    return (angle + 180.0) % 360.0 - 180.0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = p.parse_args()
    root = args.root.resolve()
    path = root / 'results' / 'benchmark_1a' / 'trials.csv'
    if not path.exists():
        raise SystemExit(f'Cannot find {path}. Run from project or use --root PATH')
    d = pd.read_csv(path)
    required = {'trial_id', 'shape', 'contact', 'initial_yaw_deg',
                'final_yaw_change_deg', 'position_error_mm', 'success'}
    missing = required - set(d.columns)
    if missing:
        raise SystemExit(f'Missing columns: {sorted(missing)}')
    if len(d) != 30 or d.trial_id.nunique() != 30:
        raise SystemExit('Expected 30 distinct trials; refusing to produce a final benchmark score.')

    d['success_original'] = d['success'].astype(int)
    d['yaw_error_original_deg'] = d['yaw_error_deg']
    d['final_world_yaw_deg'] = wrap_deg(d['initial_yaw_deg'] + d['final_yaw_change_deg'])
    # Target orientation: fixed zero yaw in the table reference frame.
    d['yaw_error_deg'] = wrap_deg(d['final_world_yaw_deg']).abs()
    position_ok = d['position_error_mm'] <= 20.0
    orientation_ok = (d['yaw_error_deg'] <= 5.0) | d['shape'].eq('cylinder')
    d['success'] = (position_ok & orientation_ok).astype(int)

    out = path.with_name('trials_fixed_world_yaw.csv')
    d.to_csv(out, index=False)
    before = int(d['success_original'].sum())
    after = int(d['success'].sum())
    changes = d[d['success_original'] != d['success']]
    print(f'Original: {before}/{len(d)} ({before/len(d):.1%})')
    print(f'Corrected: {after}/{len(d)} ({after/len(d):.1%})')
    print('Changed trial classifications:')
    if changes.empty:
        print('  None')
    else:
        print(changes[['trial_id', 'shape', 'initial_yaw_deg',
                       'final_yaw_change_deg', 'final_world_yaw_deg',
                       'position_error_mm', 'success_original', 'success']].to_string(index=False))
    print('\nSuccess by condition:')
    print(d.groupby(['shape','contact'])['success'].agg(['sum','count','mean']).to_string())

    plot_dir = root / 'media' / 'benchmark_1a'
    plot_dir.mkdir(parents=True, exist_ok=True)
    grouped = d.groupby(['shape', 'contact'])['success'].mean().unstack()
    grouped = grouped.reindex(index=['cube','rectangle','cylinder'],
                              columns=['nominal','low','patch'])
    ax = grouped.plot(kind='bar', figsize=(9,5))
    ax.set(xlabel='Geometry', ylabel='Success fraction', ylim=(0,1.05),
           title='Module 1A.4 — Success by condition (fixed world-yaw target)')
    ax.legend(title='Contact')
    ax.grid(axis='y', alpha=0.25)
    ax.figure.tight_layout()
    fig_path = plot_dir / 'success_by_condition_fixed_world_yaw.png'
    ax.figure.savefig(fig_path, dpi=170)
    plt.close(ax.figure)
    print(f'\nSaved corrected CSV: {out}')
    print(f'Saved corrected plot: {fig_path}')
    print('Raw trials.csv and existing plots were not modified.')


if __name__ == '__main__':
    main()
