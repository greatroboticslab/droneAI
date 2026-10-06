"""
Pick the held-out test videos: about 20% of the labeled videos, never trained on.

Reads the video manifest from `python LabelGUI/video_manifest.py` and writes
analysis/data/test_videos.csv (video_id column), which train_dpflow_tabular_models.py
excludes by default. The split is by video, so no clip of a test video is ever
trained on.

    python LabelGUI/make_test_split.py --dry-run
    python LabelGUI/make_test_split.py
    python LabelGUI/make_test_split.py --force     # replace an existing split

Rules: only labeled videos with at least one event; mostly FPV (--fpv-share);
every event class appears in at least one test video when the data has it,
picked from real FPV flights first.
The file is not overwritten without --force, because a test split only stays
honest if it never changes after models have been tuned against it.
"""
import argparse
import random

import pandas as pd

from repo_paths import repo_rel, resolve_path

EVENT_COLS = ["n_severe_crash", "n_land", "n_minor_crash", "n_takeoff"]  # rarest first
OUT_COLUMNS = ["video_id", "view_type", "source", "person_name"] + EVENT_COLS + ["reason"]


def pick_test_videos(manifest, share=0.2, fpv_share=0.7, seed=42):
    """Return the test rows (a DataFrame with a `reason` column)."""
    df = manifest.copy()
    for c in EVENT_COLS:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(int)
    df = df[(df["status"] == "labeled") & (df[EVENT_COLS].sum(axis=1) > 0)]
    df = df.drop_duplicates("video_id").sort_values("video_id").reset_index(drop=True)
    if df.empty:
        return df.assign(reason=[])

    target = max(1, round(len(df) * share))
    order = list(df.index)
    random.Random(seed).shuffle(order)
    is_fpv = df["view_type"].eq("FPV")
    chosen, reasons = [], {}

    # 1) Cover every event class, preferring FPV videos.
    for col in EVENT_COLS:
        if len(chosen) >= target or df.loc[chosen, col].sum() > 0:
            continue
        having = [i for i in order if df.at[i, col] > 0 and i not in chosen]
        if not having:
            continue
        is_real = df["source"].eq("Real")
        pick = next((i for i in having if is_fpv[i] and is_real[i]),
                    next((i for i in having if is_fpv[i]), having[0]))
        chosen.append(pick)
        reasons[pick] = "has " + col[2:].replace("_", "-")

    # 2) Fill up to the FPV quota, then with any view.
    fpv_target = round(target * fpv_share)
    for i in order:
        if len(chosen) >= target:
            break
        if i not in chosen and is_fpv[i] and is_fpv[chosen].sum() < fpv_target:
            chosen.append(i)
            reasons[i] = "FPV fill"
    for i in order:
        if len(chosen) >= target:
            break
        if i not in chosen:
            chosen.append(i)
            reasons[i] = "fill"

    out = df.loc[chosen].copy()
    out["reason"] = [reasons[i] for i in chosen]
    return out.sort_values("video_id").reset_index(drop=True)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--manifest", default="analysis/data/video_manifest.csv")
    p.add_argument("--out", default="analysis/data/test_videos.csv")
    p.add_argument("--share", type=float, default=0.2, help="Fraction of labeled videos to hold out")
    p.add_argument("--fpv-share", type=float, default=0.7, help="Target fraction of FPV test videos")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--force", action="store_true", help="Overwrite an existing split")
    args = p.parse_args()

    manifest_path = resolve_path(args.manifest)
    out_path = resolve_path(args.out)
    manifest = pd.read_csv(manifest_path, dtype={"video_id": str})
    test = pick_test_videos(manifest, args.share, args.fpv_share, args.seed)

    print(f"Test videos: {len(test)} | by view: {test['view_type'].value_counts().to_dict()}")
    print("Events held out:", {c[2:]: int(test[c].sum()) for c in EVENT_COLS})
    if args.dry_run:
        print(test[OUT_COLUMNS].to_string(index=False))
        return
    if out_path.exists() and not args.force:
        raise SystemExit(f"{repo_rel(out_path)} already exists. Use --force to replace the split.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    test[OUT_COLUMNS].to_csv(out_path, index=False)
    print("Wrote", repo_rel(out_path))


if __name__ == "__main__":
    main()
