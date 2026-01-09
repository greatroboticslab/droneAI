# analysis/scripts/verify_crash_counts.py
import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", default="analysis/output/per_video.csv", help="YOLO predicted per_video.csv")
    parser.add_argument("--manual", required=True, help="Manual summary CSV (person, youtube_link, manual_crashes)")
    parser.add_argument("--out", default="analysis/output")
    args = parser.parse_args()

    pred = pd.read_csv(Path(args.pred))
    pred = pred[pred["status"] == "ok"].copy()

    manual = pd.read_csv(Path(args.manual))

    # Normalize join keys
    pred["person_key"] = pred["person"].astype(str).str.strip().str.lower()
    pred["link_key"] = pred["youtube_link"].astype(str).str.strip()

    manual["person_key"] = manual["person"].astype(str).str.strip().str.lower()
    manual["link_key"] = manual["youtube_link"].astype(str).str.strip()

    # Merge
    m = pred.merge(
        manual[["person_key", "link_key", "manual_crashes"]],
        on=["person_key", "link_key"],
        how="left"
    )

    m["manual_crashes"] = pd.to_numeric(m["manual_crashes"], errors="coerce")
    m["difference"] = m["crash_events"] - m["manual_crashes"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "verification_report.csv"
    m.to_csv(out_csv, index=False)

    print("Wrote:", out_csv)
    print("Tip: look at rows where manual_crashes is NaN (missing manual labels).")


if __name__ == "__main__":
    main()

