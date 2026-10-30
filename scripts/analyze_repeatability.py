"""Analyze Isaac Sim Module 1A.2 repeatability outputs.

Run from the repository root with regular Python:
    python -m pip install numpy pandas matplotlib
    python scripts/analyze_repeatability.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
summary = pd.read_csv(RESULTS / "repeatability_summary.csv")
trajectories = pd.read_csv(RESULTS / "repeatability_trajectories.csv")
required_summary = {"trial", "dx_mm", "dy_mm", "final_yaw_deg"}
required_trajectory = {"trial", "time_s", "object_x_m", "object_y_m"}
assert required_summary.issubset(summary.columns), "Unexpected summary CSV columns"
assert required_trajectory.issubset(trajectories.columns), "Unexpected trajectory CSV columns"

# Inspect all individual motion traces; overlapping lines are expected.
fig, ax = plt.subplots(figsize=(8, 4.8))
for trial, df in trajectories.groupby("trial"):
    ax.plot(df["time_s"], df["object_x_m"] * 1000, alpha=0.65, linewidth=1.4, label=f"Trial {int(trial)}")
ax.set(title="Module 1A.2 | Object X position during push", xlabel="Time during push (s)", ylabel="Object X position (mm)")
ax.grid(alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
fig.savefig(RESULTS / "repeatability_trajectories.png", dpi=180)
plt.close(fig)

# Plot terminal displacement relative to the trial mean to resolve tiny differences.
dx = summary["dx_mm"].to_numpy(dtype=float)
mean = float(np.mean(dx))
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.plot(summary["trial"], (dx - mean) * 1000, "o-")
ax.axhline(0, linestyle="--", linewidth=1)
ax.set(title="Module 1A.2 | Final X displacement deviation", xlabel="Trial", ylabel="Deviation from mean (micrometers)")
ax.set_xticks(summary["trial"].astype(int))
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(RESULTS / "repeatability_final_deviation.png", dpi=180)
plt.close(fig)

print(f"Trials: {len(summary)}")
print(f"Mean X displacement: {mean:.9f} mm")
print(f"Sample standard deviation: {np.std(dx, ddof=1):.9f} mm")
print(f"Maximum absolute deviation: {np.max(np.abs(dx - mean)):.9f} mm")
print("Plots:", RESULTS / "repeatability_trajectories.png", "and", RESULTS / "repeatability_final_deviation.png", sep="\n  ")
