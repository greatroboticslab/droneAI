import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from inference_sdk import InferenceHTTPClient

import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"
MANIFEST_PATH = FRAME_DATASET_DIR / "frame_manifest.csv"
MOTION_RESULTS_DIR = BASE_DIR / "MotionResults"


def safe_float(value, default=np.nan):
    try:
        if value == "" or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            return img.size  # width, height
    except Exception:
        return None, None


def choose_best_detection(predictions, conf_threshold=0.25):
    """
    Pick the highest-confidence drone detection.
    If there are multiple boxes, this keeps the best one.
    """

    if not predictions:
        return None

    candidates = []

    for pred in predictions:
        class_name = str(pred.get("class", "")).lower()
        confidence = safe_float(pred.get("confidence"), default=0.0)

        # Accept Drone, drone, Drone-Detection, etc.
        if "drone" not in class_name:
            continue

        if confidence < conf_threshold:
            continue

        candidates.append(pred)

    if not candidates:
        return None

    candidates = sorted(
        candidates,
        key=lambda p: safe_float(p.get("confidence"), default=0.0),
        reverse=True,
    )

    return candidates[0]


def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Could not find {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required = {"image_path", "label", "session_name", "clip_filename"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"frame_manifest.csv is missing columns: {missing}")

    df["image_path"] = df["image_path"].astype(str)
    df["label"] = df["label"].astype(str)
    df["session_name"] = df["session_name"].astype(str)
    df["clip_filename"] = df["clip_filename"].astype(str)
    df["clip_group"] = df["session_name"] + "__" + df["clip_filename"]

    if "frame_time_sec" not in df.columns:
        df["frame_time_sec"] = np.nan

    if "source_frame_index" not in df.columns:
        df["source_frame_index"] = np.arange(len(df))

    df["frame_time_sec"] = pd.to_numeric(df["frame_time_sec"], errors="coerce")
    df["source_frame_index"] = pd.to_numeric(df["source_frame_index"], errors="coerce")

    # If frame_time_sec is missing, estimate based on frame order at 5 FPS.
    df = df.sort_values(["clip_group", "source_frame_index", "image_path"]).reset_index(drop=True)

    for clip_group, idxs in df.groupby("clip_group").groups.items():
        idxs = list(idxs)
        missing_time = df.loc[idxs, "frame_time_sec"].isna()

        if missing_time.any():
            estimated_times = np.arange(len(idxs)) / 5.0
            df.loc[idxs, "frame_time_sec"] = estimated_times

    return df


def run_detector_on_frames(df, client, model_id, output_dir, conf_threshold=0.25, max_frames=None):
    rows = []

    if max_frames:
        df = df.head(max_frames).copy()

    total = len(df)

    print(f"[INFO] Running detector on {total} frames...")

    for i, row in df.iterrows():
        image_rel = row["image_path"]
        image_path = FRAME_DATASET_DIR / image_rel

        if not image_path.exists():
            rows.append({
                "image_path": image_rel,
                "label": row["label"],
                "session_name": row["session_name"],
                "clip_filename": row["clip_filename"],
                "clip_group": row["clip_group"],
                "frame_time_sec": row["frame_time_sec"],
                "source_frame_index": row["source_frame_index"],
                "detected": False,
                "note": "image missing",
            })
            continue

        if len(rows) % 25 == 0:
            print(f"[INFO] {len(rows)}/{total}")

        try:
            result = client.infer(str(image_path), model_id=model_id)
            predictions = result.get("predictions", [])
        except Exception as e:
            rows.append({
                "image_path": image_rel,
                "label": row["label"],
                "session_name": row["session_name"],
                "clip_filename": row["clip_filename"],
                "clip_group": row["clip_group"],
                "frame_time_sec": row["frame_time_sec"],
                "source_frame_index": row["source_frame_index"],
                "detected": False,
                "note": f"inference error: {e}",
            })
            continue

        best = choose_best_detection(predictions, conf_threshold=conf_threshold)

        image_w, image_h = get_image_size(image_path)

        if best is None:
            rows.append({
                "image_path": image_rel,
                "label": row["label"],
                "session_name": row["session_name"],
                "clip_filename": row["clip_filename"],
                "clip_group": row["clip_group"],
                "frame_time_sec": row["frame_time_sec"],
                "source_frame_index": row["source_frame_index"],
                "image_width": image_w,
                "image_height": image_h,
                "detected": False,
                "class": "",
                "confidence": "",
                "x": "",
                "y": "",
                "width": "",
                "height": "",
                "bbox_area": "",
                "note": "no detection",
            })
            continue

        x = safe_float(best.get("x"))
        y = safe_float(best.get("y"))
        w = safe_float(best.get("width"))
        h = safe_float(best.get("height"))
        confidence = safe_float(best.get("confidence"))

        rows.append({
            "image_path": image_rel,
            "label": row["label"],
            "session_name": row["session_name"],
            "clip_filename": row["clip_filename"],
            "clip_group": row["clip_group"],
            "frame_time_sec": row["frame_time_sec"],
            "source_frame_index": row["source_frame_index"],
            "image_width": image_w,
            "image_height": image_h,
            "detected": True,
            "class": best.get("class", ""),
            "confidence": confidence,
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "bbox_area": w * h if not np.isnan(w) and not np.isnan(h) else "",
            "note": "",
        })

    frame_df = pd.DataFrame(rows)

    output_path = output_dir / "frame_detections.csv"
    frame_df.to_csv(output_path, index=False)

    print(f"[INFO] Frame detections saved to: {output_path}")

    return frame_df


def compute_clip_features(frame_df):
    feature_rows = []

    for clip_group, group in frame_df.groupby("clip_group"):
        group = group.copy()
        group = group.sort_values(["frame_time_sec", "source_frame_index", "image_path"])

        label = group["label"].iloc[0]
        session_name = group["session_name"].iloc[0]
        clip_filename = group["clip_filename"].iloc[0]

        total_frames = len(group)

        det = group[group["detected"] == True].copy()
        detected_frames = len(det)

        detection_rate = detected_frames / total_frames if total_frames else 0.0

        base_row = {
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_filename,
            "label": label,
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "missing_frames": total_frames - detected_frames,
            "detection_rate": detection_rate,
        }

        if detected_frames == 0:
            feature_rows.append({
                **base_row,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "dx_total": 0.0,
                "dy_total": 0.0,
                "straight_line_displacement": 0.0,
                "path_length": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "std_speed": 0.0,
                "mean_vx": 0.0,
                "mean_vy": 0.0,
                "max_downward_vy": 0.0,
                "max_upward_vy": 0.0,
                "mean_accel": 0.0,
                "max_accel": 0.0,
                "bbox_area_mean": 0.0,
                "bbox_area_start": 0.0,
                "bbox_area_end": 0.0,
                "bbox_area_change": 0.0,
            })
            continue

        det["x"] = pd.to_numeric(det["x"], errors="coerce")
        det["y"] = pd.to_numeric(det["y"], errors="coerce")
        det["confidence"] = pd.to_numeric(det["confidence"], errors="coerce")
        det["bbox_area"] = pd.to_numeric(det["bbox_area"], errors="coerce")
        det["frame_time_sec"] = pd.to_numeric(det["frame_time_sec"], errors="coerce")

        det = det.dropna(subset=["x", "y", "frame_time_sec"])

        if len(det) == 0:
            feature_rows.append({
                **base_row,
                "mean_confidence": 0.0,
                "min_confidence": 0.0,
                "max_confidence": 0.0,
                "start_x": np.nan,
                "start_y": np.nan,
                "end_x": np.nan,
                "end_y": np.nan,
                "dx_total": 0.0,
                "dy_total": 0.0,
                "straight_line_displacement": 0.0,
                "path_length": 0.0,
                "mean_speed": 0.0,
                "max_speed": 0.0,
                "std_speed": 0.0,
                "mean_vx": 0.0,
                "mean_vy": 0.0,
                "max_downward_vy": 0.0,
                "max_upward_vy": 0.0,
                "mean_accel": 0.0,
                "max_accel": 0.0,
                "bbox_area_mean": 0.0,
                "bbox_area_start": 0.0,
                "bbox_area_end": 0.0,
                "bbox_area_change": 0.0,
            })
            continue

        xs = det["x"].to_numpy(dtype=float)
        ys = det["y"].to_numpy(dtype=float)
        ts = det["frame_time_sec"].to_numpy(dtype=float)

        start_x, start_y = xs[0], ys[0]
        end_x, end_y = xs[-1], ys[-1]

        dx_total = end_x - start_x
        dy_total = end_y - start_y

        straight_line_displacement = float(np.sqrt(dx_total ** 2 + dy_total ** 2))

        speeds = []
        vxs = []
        vys = []
        segment_lengths = []

        if len(det) >= 2:
            dx = np.diff(xs)
            dy = np.diff(ys)
            dt = np.diff(ts)

            valid = dt > 0

            dx = dx[valid]
            dy = dy[valid]
            dt = dt[valid]

            if len(dt) > 0:
                segment_lengths = np.sqrt(dx ** 2 + dy ** 2)
                speeds = segment_lengths / dt
                vxs = dx / dt
                vys = dy / dt

        speeds = np.array(speeds, dtype=float)
        vxs = np.array(vxs, dtype=float)
        vys = np.array(vys, dtype=float)
        segment_lengths = np.array(segment_lengths, dtype=float)

        if len(speeds) >= 2:
            accel = np.abs(np.diff(speeds))
        else:
            accel = np.array([], dtype=float)

        bbox_area = det["bbox_area"].dropna().to_numpy(dtype=float)

        bbox_area_start = float(bbox_area[0]) if len(bbox_area) else 0.0
        bbox_area_end = float(bbox_area[-1]) if len(bbox_area) else 0.0

        # Positive y direction means moving downward in image coordinates.
        feature_rows.append({
            **base_row,
            "mean_confidence": float(det["confidence"].mean()) if det["confidence"].notna().any() else 0.0,
            "min_confidence": float(det["confidence"].min()) if det["confidence"].notna().any() else 0.0,
            "max_confidence": float(det["confidence"].max()) if det["confidence"].notna().any() else 0.0,
            "start_x": float(start_x),
            "start_y": float(start_y),
            "end_x": float(end_x),
            "end_y": float(end_y),
            "dx_total": float(dx_total),
            "dy_total": float(dy_total),
            "straight_line_displacement": straight_line_displacement,
            "path_length": float(segment_lengths.sum()) if len(segment_lengths) else 0.0,
            "mean_speed": float(speeds.mean()) if len(speeds) else 0.0,
            "max_speed": float(speeds.max()) if len(speeds) else 0.0,
            "std_speed": float(speeds.std()) if len(speeds) else 0.0,
            "mean_vx": float(vxs.mean()) if len(vxs) else 0.0,
            "mean_vy": float(vys.mean()) if len(vys) else 0.0,
            "max_downward_vy": float(vys.max()) if len(vys) else 0.0,
            "max_upward_vy": float(vys.min()) if len(vys) else 0.0,
            "mean_accel": float(accel.mean()) if len(accel) else 0.0,
            "max_accel": float(accel.max()) if len(accel) else 0.0,
            "bbox_area_mean": float(bbox_area.mean()) if len(bbox_area) else 0.0,
            "bbox_area_start": bbox_area_start,
            "bbox_area_end": bbox_area_end,
            "bbox_area_change": bbox_area_end - bbox_area_start,
        })

    return pd.DataFrame(feature_rows)


def save_trajectory_plots(frame_df, output_dir, max_plots=100):
    plot_dir = output_dir / "trajectory_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    saved = 0

    for clip_group, group in frame_df.groupby("clip_group"):
        if saved >= max_plots:
            break

        group = group.copy()
        group = group[group["detected"] == True]
        group = group.sort_values(["frame_time_sec", "source_frame_index", "image_path"])

        if len(group) < 2:
            continue

        xs = pd.to_numeric(group["x"], errors="coerce")
        ys = pd.to_numeric(group["y"], errors="coerce")

        valid = xs.notna() & ys.notna()

        xs = xs[valid]
        ys = ys[valid]

        if len(xs) < 2:
            continue

        label = group["label"].iloc[0]
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in clip_group)
        out_path = plot_dir / f"{label}__{safe_name}.png"

        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, marker="o")
        plt.gca().invert_yaxis()
        plt.title(f"{label}\n{clip_group}")
        plt.xlabel("x center")
        plt.ylabel("y center")
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

        saved += 1

    print(f"[INFO] Saved {saved} trajectory plots to: {plot_dir}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-id", required=True, help="Roboflow model id, example: drone-detection/2")
    parser.add_argument("--api-key", default="", help="Roboflow API key. Or set ROBOFLOW_API_KEY.")
    parser.add_argument("--run-name", default="motion_v2")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--max-plots", type=int, default=100)

    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.")

    output_dir = MOTION_RESULTS_DIR / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_manifest()

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    frame_df = run_detector_on_frames(
        df=df,
        client=client,
        model_id=args.model_id,
        output_dir=output_dir,
        conf_threshold=args.conf,
        max_frames=args.max_frames if args.max_frames > 0 else None,
    )

    features_df = compute_clip_features(frame_df)

    features_path = output_dir / "clip_motion_features.csv"
    features_df.to_csv(features_path, index=False)

    save_trajectory_plots(
        frame_df=frame_df,
        output_dir=output_dir,
        max_plots=args.max_plots,
    )

    summary = {
        "run_name": args.run_name,
        "model_id": args.model_id,
        "confidence_threshold": args.conf,
        "total_frames": int(len(frame_df)),
        "detected_frames": int((frame_df["detected"] == True).sum()),
        "frame_detection_rate": float((frame_df["detected"] == True).mean()),
        "total_clips": int(features_df["clip_group"].nunique()),
        "features_csv": str(features_path),
        "frame_detections_csv": str(output_dir / "frame_detections.csv"),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nDone.")
    print(f"Frame detections: {output_dir / 'frame_detections.csv'}")
    print(f"Clip motion features: {features_path}")
    print(f"Run summary: {output_dir / 'run_summary.json'}")
    print(f"Frame detection rate: {summary['frame_detection_rate']:.4f}")


if __name__ == "__main__":
    main()
