# analysis/scripts/summarize_and_correlate.py
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per_video", default="analysis/output/per_video.csv")
    parser.add_argument("--out", default="analysis/output")
    args = parser.parse_args()

    per_video_path = Path(args.per_video)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(per_video_path)

    # Keep only successful rows
    df = df[df["status"] == "ok"].copy()

    # Normalize scenario names
    df["scenario_norm"] = df["scenario"].astype(str).str.strip().str.lower()
    df["scenario_norm"] = df["scenario_norm"].replace({
        "simulation": "simulation",
        "sim": "simulation",
        "real flight": "real",
        "real": "real",
        "realflight": "real",
    })

    # Per-person summary by scenario
    g = df.groupby(["person", "scenario_norm"], as_index=False).agg(
        videos=("youtube_link", "count"),
        crashes=("crash_events", "sum"),
        avg_crashes_per_min=("crashes_per_min", "mean"),
    )

    summary_path = out_dir / "per_person_summary.csv"
    g.to_csv(summary_path, index=False)

    # Pivot for correlation (simulation vs real)
    pivot = g.pivot_table(index="person", columns="scenario_norm", values="avg_crashes_per_min", aggfunc="mean")
    pivot = pivot.reset_index()

    # Correlation only for people with both sim and real
    both = pivot.dropna(subset=["simulation", "real"], how="any").copy()
    corr = both["simulation"].corr(both["real"]) if len(both) >= 2 else None

    corr_path = out_dir / "correlation_report.txt"
    with open(corr_path, "w", encoding="utf-8") as f:
        f.write("Crash Rate Correlation Report\n")
        f.write("=============================\n\n")
        f.write(f"People with BOTH simulation and real: {len(both)}\n\n")
        if corr is None:
            f.write("Not enough paired data to compute correlation.\n")
        else:
            f.write(f"Pearson correlation (avg crashes/min): {corr:.4f}\n")

    print("Wrote:")
    print(" ", summary_path)
    print(" ", corr_path)


if __name__ == "__main__":
    main()

