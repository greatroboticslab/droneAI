import argparse
import json
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"
FRAME_MANIFEST_PATH = FRAME_DATASET_DIR / "frame_manifest.csv"

DEFAULT_DETECTIONS_PATH = (
    BASE_DIR / "MotionResults" / "motion_v2_all" / "frame_detections.csv"
)

OPTICAL_FLOW_RESULTS_DIR = BASE_DIR / "OpticalFlowResults"


def to_bool(value):
    return str(value).strip().lower() in ["true", "1", "yes", "y"]


def safe_float(value, default=np.nan):
    try:
        if value == "" or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def load_inputs(detections_csv):
    if not FRAME_MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Could not find {FRAME_MANIFEST_PATH}")

    if not detections_csv.exists():
        raise FileNotFoundError(f"Could not find {detections_csv}")

    manifest = pd.read_csv(FRAME_MANIFEST_PATH)
    detections = pd.read_csv(detections_csv)

    required_manifest = {"image_path", "label", "session_name", "clip_filename"}
    missing_manifest = required_manifest - set(manifest.columns)

    if missing_manifest:
        raise ValueError(f"frame_manifest.csv missing columns: {missing_manifest}")

    required_detections = {"image_path", "detected", "x", "y", "width", "height", "confidence"}
    missing_detections = required_detections - set(detections.columns)

    if missing_detections:
        raise ValueError(f"frame_detections.csv missing columns: {missing_detections}")

    manifest["image_path"] = manifest["image_path"].astype(str)
    manifest["label"] = manifest["label"].astype(str)
    manifest["session_name"] = manifest["session_name"].astype(str)
    manifest["clip_filename"] = manifest["clip_filename"].astype(str)

    if "frame_time_sec" not in manifest.columns:
        manifest["frame_time_sec"] = np.nan

    if "source_frame_index" not in manifest.columns:
        manifest["source_frame_index"] = np.nan

    manifest["frame_time_sec"] = pd.to_numeric(manifest["frame_time_sec"], errors="coerce")
    manifest["source_frame_index"] = pd.to_numeric(manifest["source_frame_index"], errors="coerce")

    manifest["clip_group"] = manifest["session_name"] + "__" + manifest["clip_filename"]

    # Fill missing frame times based on order, assuming 5 FPS extraction.
    manifest = manifest.sort_values(
        ["clip_group", "source_frame_index", "image_path"]
    ).reset_index(drop=True)

    for clip_group, idxs in manifest.groupby("clip_group").groups.items():
        idxs = list(idxs)
        if manifest.loc[idxs, "frame_time_sec"].isna().any():
            manifest.loc[idxs, "frame_time_sec"] = np.arange(len(idxs)) / 5.0

    detections["image_path"] = detections["image_path"].astype(str)
    detections["detected_bool"] = detections["detected"].apply(to_bool)

    # One detection row per image, because our previous detector script already chose the best box.
    detections = detections.drop_duplicates(subset=["image_path"], keep="first")

    det_map = {
        row["image_path"]: row
        for _, row in detections.iterrows()
    }

    return manifest, det_map


def read_gray(image_rel_path):
    image_path = FRAME_DATASET_DIR / image_rel_path

    if not image_path.exists():
        return None, None

    image = cv2.imread(str(image_path))

    if image is None:
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray


def get_bbox(det_row):
    if det_row is None:
        return None

    if not to_bool(det_row.get("detected_bool", det_row.get("detected", False))):
        return None

    x = safe_float(det_row.get("x"))
    y = safe_float(det_row.get("y"))
    w = safe_float(det_row.get("width"))
    h = safe_float(det_row.get("height"))

    if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
        return None

    if w <= 0 or h <= 0:
        return None

    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0

    return {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "confidence": safe_float(det_row.get("confidence"), default=0.0),
    }


def make_roi(boxes, image_w, image_h, min_size=96, pad_factor=2.5):
    """
    Build one crop region that covers the detected drone box in frame t and t+1.
    This keeps optical flow focused around the drone.
    """

    boxes = [b for b in boxes if b is not None]

    if not boxes:
        return None

    x1 = min(b["x1"] for b in boxes)
    y1 = min(b["y1"] for b in boxes)
    x2 = max(b["x2"] for b in boxes)
    y2 = max(b["y2"] for b in boxes)

    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    roi_w = max(min_size, box_w * pad_factor)
    roi_h = max(min_size, box_h * pad_factor)

    rx1 = int(round(cx - roi_w / 2.0))
    ry1 = int(round(cy - roi_h / 2.0))
    rx2 = int(round(cx + roi_w / 2.0))
    ry2 = int(round(cy + roi_h / 2.0))

    rx1 = max(0, rx1)
    ry1 = max(0, ry1)
    rx2 = min(image_w, rx2)
    ry2 = min(image_h, ry2)

    if rx2 <= rx1 + 5 or ry2 <= ry1 + 5:
        return None

    return rx1, ry1, rx2, ry2


def compute_farneback_flow(prev_gray, next_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        next_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    dx = flow[..., 0]
    dy = flow[..., 1]

    mag, ang = cv2.cartToPolar(dx, dy, angleInDegrees=False)

    return {
        "flow_dx_mean": float(np.mean(dx)),
        "flow_dy_mean": float(np.mean(dy)),
        "flow_dx_median": float(np.median(dx)),
        "flow_dy_median": float(np.median(dy)),
        "flow_mag_mean": float(np.mean(mag)),
        "flow_mag_median": float(np.median(mag)),
        "flow_mag_max": float(np.max(mag)),
        "flow_mag_std": float(np.std(mag)),
        "flow_angle_mean": float(np.mean(ang)),
    }


def draw_debug_image(prev_image, row, roi, bbox_a, bbox_b, out_path):
    image = prev_image.copy()

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    if bbox_a is not None:
        x1 = int(bbox_a["x1"])
        y1 = int(bbox_a["y1"])
        x2 = int(bbox_a["x2"])
        y2 = int(bbox_a["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if bbox_b is not None:
        x1 = int(bbox_b["x1"])
        y1 = int(bbox_b["y1"])
        x2 = int(bbox_b["x2"])
        y2 = int(bbox_b["y2"])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 255), 2)

    if bbox_a is not None and bbox_b is not None:
        p1 = (int(bbox_a["x"]), int(bbox_a["y"]))
        p2 = (int(bbox_b["x"]), int(bbox_b["y"]))
        cv2.arrowedLine(image, p1, p2, (0, 0, 255), 2, tipLength=0.3)

    label = f"{row.get('label', '')} step {row.get('step_index', '')}"
    cv2.putText(
        image,
        label,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def extract_flow_sequences(manifest, det_map, output_dir, max_debug=80):
    rows = []
    debug_count = 0
    debug_dir = output_dir / "debug_flow_images"

    total_clips = manifest["clip_group"].nunique()

    print(f"[INFO] Clips found: {total_clips}")

    for clip_idx, (clip_group, group) in enumerate(manifest.groupby("clip_group"), start=1):
        group = group.copy()
        group = group.sort_values(["frame_time_sec", "source_frame_index", "image_path"])

        if len(group) < 2:
            continue

        label = group["label"].iloc[0]
        session_name = group["session_name"].iloc[0]
        clip_filename = group["clip_filename"].iloc[0]

        if clip_idx % 10 == 0:
            print(f"[INFO] Processing clip {clip_idx}/{total_clips}: {clip_group}")

        group_rows = group.to_dict("records")

        for step_index in range(len(group_rows) - 1):
            a = group_rows[step_index]
            b = group_rows[step_index + 1]

            image_a_rel = a["image_path"]
            image_b_rel = b["image_path"]

            image_a, gray_a = read_gray(image_a_rel)
            image_b, gray_b = read_gray(image_b_rel)

            if gray_a is None or gray_b is None:
                continue

            image_h, image_w = gray_a.shape[:2]

            det_a = det_map.get(image_a_rel)
            det_b = det_map.get(image_b_rel)

            bbox_a = get_bbox(det_a)
            bbox_b = get_bbox(det_b)

            detected_a = bbox_a is not None
            detected_b = bbox_b is not None

            t0 = safe_float(a.get("frame_time_sec"), default=step_index / 5.0)
            t1 = safe_float(b.get("frame_time_sec"), default=(step_index + 1) / 5.0)
            dt = max(1e-6, t1 - t0)

            roi = make_roi([bbox_a, bbox_b], image_w, image_h)

            base = {
                "clip_group": clip_group,
                "session_name": session_name,
                "clip_filename": clip_filename,
                "label": label,
                "step_index": step_index,
                "from_image": image_a_rel,
                "to_image": image_b_rel,
                "t0": t0,
                "t1": t1,
                "dt": dt,
                "image_width": image_w,
                "image_height": image_h,
                "detected_a": detected_a,
                "detected_b": detected_b,
                "both_detected": detected_a and detected_b,
                "any_detected": detected_a or detected_b,
            }

            if roi is None:
                row = {
                    **base,
                    "roi_available": False,
                    "roi_x1": "",
                    "roi_y1": "",
                    "roi_x2": "",
                    "roi_y2": "",
                    "roi_width": 0,
                    "roi_height": 0,
                    "flow_dx_mean": 0.0,
                    "flow_dy_mean": 0.0,
                    "flow_dx_median": 0.0,
                    "flow_dy_median": 0.0,
                    "flow_mag_mean": 0.0,
                    "flow_mag_median": 0.0,
                    "flow_mag_max": 0.0,
                    "flow_mag_std": 0.0,
                    "flow_angle_mean": 0.0,
                    "flow_dx_mean_per_sec": 0.0,
                    "flow_dy_mean_per_sec": 0.0,
                    "flow_mag_mean_per_sec": 0.0,
                    "flow_dx_norm_per_sec": 0.0,
                    "flow_dy_norm_per_sec": 0.0,
                    "flow_mag_norm_per_sec": 0.0,
                    "det_dx": 0.0,
                    "det_dy": 0.0,
                    "det_speed": 0.0,
                    "det_vx_per_sec": 0.0,
                    "det_vy_per_sec": 0.0,
                    "det_speed_per_sec": 0.0,
                    "det_vx_norm_per_sec": 0.0,
                    "det_vy_norm_per_sec": 0.0,
                    "det_speed_norm_per_sec": 0.0,
                    "conf_a": bbox_a["confidence"] if bbox_a is not None else 0.0,
                    "conf_b": bbox_b["confidence"] if bbox_b is not None else 0.0,
                }

                rows.append(row)
                continue

            rx1, ry1, rx2, ry2 = roi

            crop_a = gray_a[ry1:ry2, rx1:rx2]
            crop_b = gray_b[ry1:ry2, rx1:rx2]

            flow_stats = compute_farneback_flow(crop_a, crop_b)

            det_dx = 0.0
            det_dy = 0.0
            det_speed = 0.0
            det_vx_per_sec = 0.0
            det_vy_per_sec = 0.0
            det_speed_per_sec = 0.0

            if bbox_a is not None and bbox_b is not None:
                det_dx = bbox_b["x"] - bbox_a["x"]
                det_dy = bbox_b["y"] - bbox_a["y"]
                det_speed = float(np.sqrt(det_dx ** 2 + det_dy ** 2))

                det_vx_per_sec = det_dx / dt
                det_vy_per_sec = det_dy / dt
                det_speed_per_sec = det_speed / dt

            flow_dx_mean_per_sec = flow_stats["flow_dx_mean"] / dt
            flow_dy_mean_per_sec = flow_stats["flow_dy_mean"] / dt
            flow_mag_mean_per_sec = flow_stats["flow_mag_mean"] / dt

            row = {
                **base,
                "roi_available": True,
                "roi_x1": rx1,
                "roi_y1": ry1,
                "roi_x2": rx2,
                "roi_y2": ry2,
                "roi_width": rx2 - rx1,
                "roi_height": ry2 - ry1,

                **flow_stats,

                "flow_dx_mean_per_sec": flow_dx_mean_per_sec,
                "flow_dy_mean_per_sec": flow_dy_mean_per_sec,
                "flow_mag_mean_per_sec": flow_mag_mean_per_sec,

                "flow_dx_norm_per_sec": flow_dx_mean_per_sec / image_w,
                "flow_dy_norm_per_sec": flow_dy_mean_per_sec / image_h,
                "flow_mag_norm_per_sec": flow_mag_mean_per_sec / max(image_w, image_h),

                "det_dx": det_dx,
                "det_dy": det_dy,
                "det_speed": det_speed,
                "det_vx_per_sec": det_vx_per_sec,
                "det_vy_per_sec": det_vy_per_sec,
                "det_speed_per_sec": det_speed_per_sec,

                "det_vx_norm_per_sec": det_vx_per_sec / image_w,
                "det_vy_norm_per_sec": det_vy_per_sec / image_h,
                "det_speed_norm_per_sec": det_speed_per_sec / max(image_w, image_h),

                "conf_a": bbox_a["confidence"] if bbox_a is not None else 0.0,
                "conf_b": bbox_b["confidence"] if bbox_b is not None else 0.0,
            }

            rows.append(row)

            if debug_count < max_debug:
                safe_clip = "".join(c if c.isalnum() or c in "-_." else "_" for c in clip_group)
                debug_path = debug_dir / f"{label}__{safe_clip}__step_{step_index:03d}.jpg"
                draw_debug_image(image_a, row, roi, bbox_a, bbox_b, debug_path)
                debug_count += 1

    flow_df = pd.DataFrame(rows)
    return flow_df


def summarize_clip_features(flow_df):
    summary_rows = []

    for clip_group, group in flow_df.groupby("clip_group"):
        group = group.copy()

        label = group["label"].iloc[0]
        session_name = group["session_name"].iloc[0]
        clip_filename = group["clip_filename"].iloc[0]

        numeric_cols = [
            "flow_dx_norm_per_sec",
            "flow_dy_norm_per_sec",
            "flow_mag_norm_per_sec",
            "flow_mag_mean",
            "flow_mag_max",
            "det_vx_norm_per_sec",
            "det_vy_norm_per_sec",
            "det_speed_norm_per_sec",
            "det_speed",
            "conf_a",
            "conf_b",
        ]

        row = {
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_filename,
            "label": label,
            "steps": len(group),
            "roi_available_rate": float(group["roi_available"].mean()),
            "both_detected_rate": float(group["both_detected"].mean()),
            "any_detected_rate": float(group["any_detected"].mean()),
        }

        for col in numeric_cols:
            values = pd.to_numeric(group[col], errors="coerce").fillna(0)

            row[f"{col}_mean"] = float(values.mean())
            row[f"{col}_max"] = float(values.max())
            row[f"{col}_min"] = float(values.min())
            row[f"{col}_std"] = float(values.std()) if len(values) > 1 else 0.0

        # Direction-specific useful features.
        dy = pd.to_numeric(group["flow_dy_norm_per_sec"], errors="coerce").fillna(0)
        det_vy = pd.to_numeric(group["det_vy_norm_per_sec"], errors="coerce").fillna(0)

        # Positive y means downward in image coordinates.
        row["max_downward_flow"] = float(dy.max())
        row["max_upward_flow"] = float(dy.min())
        row["mean_downward_flow"] = float(dy[dy > 0].mean()) if (dy > 0).any() else 0.0

        row["max_downward_det_vy"] = float(det_vy.max())
        row["max_upward_det_vy"] = float(det_vy.min())
        row["mean_downward_det_vy"] = float(det_vy[det_vy > 0].mean()) if (det_vy > 0).any() else 0.0

        speed = pd.to_numeric(group["det_speed_norm_per_sec"], errors="coerce").fillna(0)

        if len(speed) >= 2:
            accel = speed.diff().fillna(0).abs()
            row["det_accel_mean"] = float(accel.mean())
            row["det_accel_max"] = float(accel.max())
        else:
            row["det_accel_mean"] = 0.0
            row["det_accel_max"] = 0.0

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--detections-csv",
        default=str(DEFAULT_DETECTIONS_PATH),
        help="Path to frame_detections.csv from Roboflow detector run.",
    )

    parser.add_argument(
        "--run-name",
        default="flow_v1",
    )

    parser.add_argument(
        "--max-debug",
        type=int,
        default=80,
        help="Number of debug images to save.",
    )

    args = parser.parse_args()

    output_dir = OPTICAL_FLOW_RESULTS_DIR / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    detections_csv = Path(args.detections_csv)

    print("\n==============================")
    print("DroneAI Optical Flow Extractor")
    print("==============================\n")

    print(f"[INFO] Frame manifest: {FRAME_MANIFEST_PATH}")
    print(f"[INFO] Detections CSV: {detections_csv}")
    print(f"[INFO] Output folder: {output_dir}")

    manifest, det_map = load_inputs(detections_csv)

    flow_df = extract_flow_sequences(
        manifest=manifest,
        det_map=det_map,
        output_dir=output_dir,
        max_debug=args.max_debug,
    )

    sequence_path = output_dir / "flow_sequence_features.csv"
    flow_df.to_csv(sequence_path, index=False)

    clip_summary = summarize_clip_features(flow_df)
    clip_summary_path = output_dir / "clip_flow_summary.csv"
    clip_summary.to_csv(clip_summary_path, index=False)

    run_summary = {
        "created_at": datetime.now().isoformat(),
        "run_name": args.run_name,
        "total_clips": int(clip_summary["clip_group"].nunique()) if not clip_summary.empty else 0,
        "total_flow_steps": int(len(flow_df)),
        "mean_roi_available_rate": float(flow_df["roi_available"].mean()) if not flow_df.empty else 0.0,
        "mean_both_detected_rate": float(flow_df["both_detected"].mean()) if not flow_df.empty else 0.0,
        "sequence_features_csv": str(sequence_path),
        "clip_summary_csv": str(clip_summary_path),
        "debug_images": str(output_dir / "debug_flow_images"),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print("\nDone.")
    print(f"Flow sequence features: {sequence_path}")
    print(f"Clip flow summary: {clip_summary_path}")
    print(f"Run summary: {output_dir / 'run_summary.json'}")
    print(f"Debug images: {output_dir / 'debug_flow_images'}")
    print(f"Mean ROI available rate: {run_summary['mean_roi_available_rate']:.4f}")
    print(f"Mean both-detected rate: {run_summary['mean_both_detected_rate']:.4f}")


if __name__ == "__main__":
    main()
