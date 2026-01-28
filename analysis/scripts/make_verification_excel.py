"""
Create an Excel file that compares YOLO-predicted crash counts vs manual verification.

Input files (created by the pipeline + GUI):
  - analysis/results/per_video.csv   (predicted)
  - analysis/results/verified.csv    (manual verification)

Output:
  - analysis/results/crash_verification.xlsx

Run:
  python analysis/scripts/make_verification_excel.py
  # or specify paths:
  python analysis/scripts/make_verification_excel.py --results analysis/results
"""

from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="analysis/results", help="Results folder (default: analysis/results)")
    args = ap.parse_args()

    results_dir = Path(args.results)
    per_video = _read_csv(results_dir / "per_video.csv")
    verified = _read_csv(results_dir / "verified.csv")

    # Standardize column names
    if not per_video.empty:
        per_video.columns = [c.strip() for c in per_video.columns]
    if not verified.empty:
        verified.columns = [c.strip() for c in verified.columns]

    # Rename for clarity
    if "crash_events" in per_video.columns:
        per_video = per_video.rename(columns={"crash_events": "predicted_crashes"})
    if "crashes_per_min" in per_video.columns:
        per_video = per_video.rename(columns={"crashes_per_min": "predicted_per_min"})

    if "verified_crash_events" in verified.columns:
        verified = verified.rename(columns={"verified_crash_events": "verified_crashes"})
    if "verified_crashes_per_min" in verified.columns:
        verified = verified.rename(columns={"verified_crashes_per_min": "verified_per_min"})

    # Ensure merge keys exist
    for c in ["person", "scenario", "youtube_link"]:
        if c not in per_video.columns:
            per_video[c] = None
        if c not in verified.columns:
            verified[c] = None

    comparison = per_video.merge(
        verified[["person", "scenario", "youtube_link", "verified_crashes", "verified_per_min", "notes"]]
        if not verified.empty else verified,
        on=["person", "scenario", "youtube_link"],
        how="outer",
        suffixes=("_pred", "_ver"),
    )

    # Nice column order if available
    ordered = [c for c in [
        "person", "scenario", "youtube_link",
        "duration_sec",
        "predicted_crashes", "predicted_per_min",
        "verified_crashes", "verified_per_min",
        "status", "video_path", "notes"
    ] if c in comparison.columns]
    if ordered:
        comparison = comparison[ordered]

    out_xlsx = results_dir / "crash_verification.xlsx"
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as writer:
        comparison.to_excel(writer, index=False, sheet_name="comparison")
        if not per_video.empty:
            per_video.to_excel(writer, index=False, sheet_name="predicted_raw")
        if not verified.empty:
            verified.to_excel(writer, index=False, sheet_name="verified_raw")

    print("Wrote:", out_xlsx)


if __name__ == "__main__":
    main()
