"""Analyze a CSV of completed trial outcomes."""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", help="Trial CSV containing geometry, condition, success, position_error_m, yaw_error_deg")
    args = parser.parse_args()
    df = pd.read_csv(args.csv)
    required = {"geometry", "condition", "success", "position_error_m", "yaw_error_deg"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    print(df.groupby(["geometry", "condition"]).agg(trials=("success", "size"), success_rate=("success", "mean"), mean_position_error_m=("position_error_m", "mean"), mean_yaw_error_deg=("yaw_error_deg", "mean")).to_string())

if __name__ == "__main__":
    main()
