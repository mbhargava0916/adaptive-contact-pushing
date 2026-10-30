"""Plot Module 1A.3 CSVs using regular system Python."""
from pathlib import Path
import csv
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
path = root / 'results' / 'friction_comparison_trajectories.csv'
with path.open(newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

fig, ax = plt.subplots(figsize=(9, 5))
for label in ('low', 'nominal', 'high'):
    points = [r for r in rows if r['condition'] == label]
    initial_x = float(points[0]['object_x_m'])
    t = [float(r['time_s']) for r in points]
    dx = [(float(r['object_x_m']) - initial_x) * 1000 for r in points]
    ax.plot(t, dx, label=label)
ax.set(xlabel='Time since push begins (s)', ylabel='X displacement (mm)',
       title='Module 1A.3 | Cube displacement under three surface frictions')
ax.grid(alpha=0.3)
ax.legend(title='Surface condition')
fig.tight_layout()
output = root / 'results' / 'friction_comparison.png'
fig.savefig(output, dpi=180)
print(f'Saved: {output}')
plt.show()
