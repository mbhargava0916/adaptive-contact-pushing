"""Plot Module 1A.4 experiment output using ordinary Python (not Isaac Sim)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

out = Path(__file__).resolve().parents[1] / 'results' / 'benchmark_1a'
media = Path(__file__).resolve().parents[1] / 'media' / 'benchmark_1a'
media.mkdir(parents=True, exist_ok=True)
summary = pd.read_csv(out / 'trials.csv')
traj = pd.read_csv(out / 'trajectories.csv')
assert len(summary) == 30, f'Expected 30 trials, got {len(summary)}'

# Position error per trial; shades follow Matplotlib defaults.
fig,ax=plt.subplots(figsize=(12,5))
for i,(shape,group) in enumerate(summary.groupby('shape',sort=False)):
    ax.scatter(group['trial_id'],group['position_error_mm'],label=shape,s=46)
ax.axhline(20,linestyle='--',color='gray',label='position tolerance 20 mm')
ax.set(xlabel='Trial ID',ylabel='Final planar position error (mm)',
       title='Module 1A.4 — Fixed push: terminal position error')
ax.tick_params(axis='x',rotation=90)
ax.legend();ax.grid(alpha=.25);fig.tight_layout()
fig.savefig(media/'terminal_errors.png',dpi=170);plt.close(fig)

agg=summary.groupby(['shape','contact'],sort=False)['success'].mean().unstack()
fig,ax=plt.subplots(figsize=(9,5))
agg.plot(kind='bar',ax=ax)
ax.set(xlabel='Geometry',ylabel='Success fraction',ylim=(0,1.05),
       title='Module 1A.4 — Success fraction by geometry and contact')
ax.legend(title='Contact');ax.grid(axis='y',alpha=.25)
fig.tight_layout();fig.savefig(media/'success_by_condition.png',dpi=170);plt.close(fig)

# Reference pass only: motion curves and release behavior.
ref=traj[traj['pass_num']==0]
fig,ax=plt.subplots(figsize=(10,5))
for (shape,contact),group in ref.groupby(['shape','contact']):
    group=group.sort_values('time_s')
    ax.plot(group['time_s'],group['displacement_x_mm'],label=f'{shape} / {contact}')
ax.set(xlabel='Elapsed simulated time (s)',ylabel='Object X displacement (mm)',
       title='Module 1A.4 — Nine reference push trajectories')
ax.legend(fontsize=8,ncol=3);ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(media/'reference_trajectories.png',dpi=170);plt.close(fig)

# Compare exact repeats against their reference trials.
look=summary.set_index('trial_id')
print('\nExact repeat comparisons (repeat - reference):')
for _,r in summary[summary['is_repeat']==True].iterrows():
    base=look.loc[r['repeat_of']]
    print(f"{r['trial_id']} vs {r['repeat_of']}: "
          f"delta dx = {r['final_dx_mm']-base['final_dx_mm']:+.6f} mm")
print('\nSuccess by condition:')
print(summary.groupby(['shape','contact'])['success'].agg(['sum','count','mean']).to_string())
print('\nSaved figures to:',media)
