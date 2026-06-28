import argparse
import shutil
import re
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"
MANIFEST_PATH = FRAME_DATASET_DIR / "frame_manifest.csv"


def safe_name(text):
    text = str(text)
    text = text.replace("\\", "/")
    text = re.sub(r"[^\w\-./]+", "_", text)
    text = text.replace("/", "__")
    text = re.sub(r"_+", "_", text)
    return text.strip("_") or "unnamed"


def pick_evenly(group, frames_per_clip):
    group = group.copy()

    if "frame_time_sec" in group.columns:
        group = group.sort_values("frame_time_sec")
    elif "source_frame_index" in group.columns:
        group = group.sort_values("source_frame_index")
    else:
        group = group.sort_values("image_path")

    n = len(group)

    if n <= frames_per_clip:
        return group

    indices = np.linspace(0, n - 1, frames_per_clip).round().astype(int)
    return group.iloc[indices]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", default="drone_detection_v1")
    parser.add_argument("--frames-per-clip", type=int, default=4)
    parser.add_argument("--clear", action="store_true")
    args = parser.parse_args()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Could not find {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required = {"image_path", "label", "session_name", "clip_filename"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"frame_manifest.csv is missing columns: {missing}")

    df["label"] = df["label"].astype(str)
    df["session_name"] = df["session_name"].astype(str)
    df["clip_filename"] = df["clip_filename"].astype(str)
    df["clip_group"] = df["session_name"] + "__" + df["clip_filename"]

    output_dir = BASE_DIR / "RoboflowDetectionUpload" / args.dataset_name
    images_dir = output_dir / "images"

    if args.clear and output_dir.exists():
        shutil.rmtree(output_dir)

    images_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []

    for clip_group, group in df.groupby("clip_group"):
        picked = pick_evenly(group, args.frames_per_clip)

        for _, row in picked.iterrows():
            src = FRAME_DATASET_DIR / row["image_path"]

            if not src.exists():
                continue

            event_label = safe_name(row["label"])
            session = safe_name(row["session_name"])
            clip = safe_name(Path(row["clip_filename"]).stem)
            frame_index = str(row.get("source_frame_index", "x"))

            out_name = f"{event_label}__{session}__{clip}__frame_{frame_index}.jpg"
            dst = images_dir / out_name

            shutil.copy2(src, dst)

            selected_rows.append({
                "upload_image": out_name,
                "event_label": row["label"],
                "session_name": row["session_name"],
                "clip_filename": row["clip_filename"],
                "clip_group": row["clip_group"],
                "source_image_path": row["image_path"],
                "source_frame_index": row.get("source_frame_index", ""),
                "frame_time_sec": row.get("frame_time_sec", ""),
            })

    selected_df = pd.DataFrame(selected_rows)
    selected_df.to_csv(output_dir / "selection_manifest.csv", index=False)

    print("\nDone.")
    print(f"Images copied to: {images_dir}")
    print(f"Selection manifest: {output_dir / 'selection_manifest.csv'}")
    print(f"Total images selected: {len(selected_df)}")

    print("\nImages by event label:")
    print(selected_df["event_label"].value_counts())


if __name__ == "__main__":
    main()
