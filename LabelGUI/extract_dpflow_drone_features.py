import argparse
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent


def now_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_col(df, candidates, required=False, label="column"):
    lower_map = {c.lower(): c for c in df.columns}

    for name in candidates:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if required:
        raise ValueError(
            f"Could not find required {label}. Tried: {candidates}\n"
            f"Available columns: {list(df.columns)}"
        )

    return None


def resolve_path(value, base_candidates):
    if pd.isna(value):
        return None

    p = Path(str(value).strip().replace("\\", "/"))

    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.extend([
            Path.cwd() / p,
            PROJECT_DIR / p,
            BASE_DIR / p,
            *[b / p for b in base_candidates],
        ])

    for c in candidates:
        if c.exists():
            return c.resolve()

    # Return best guess even if missing, so logs are understandable.
    return candidates[0].resolve()


def parse_frame_number(path_or_name):
    text = str(path_or_name)
    nums = re.findall(r"\d+", text)
    if not nums:
        return 0
    return int(nums[-1])


def safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def load_manifest(manifest_path):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT_DIR / manifest_path

    if not manifest_path.exists():
        raise FileNotFoundError(f"Frame manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)

    path_col = find_col(
        df,
        [
            "frame_path", "image_path", "saved_frame_path", "frame_file",
            "image_file", "filepath", "path"
        ],
        required=True,
        label="frame path column",
    )

    label_col = find_col(
        df,
        ["label", "event_label", "event", "class", "class_name", "event_type"],
        required=True,
        label="label column",
    )

    session_col = find_col(
        df,
        ["session_name", "session", "source_session", "output_folder", "dataset_name"],
        required=False,
    )

    clip_col = find_col(
        df,
        ["clip_group", "clip_filename", "clip_name", "clip_id", "source_clip", "video_clip"],
        required=False,
    )

    frame_index_col = find_col(
        df,
        ["frame_index", "frame_idx", "frame_number", "frame_num", "frame_id"],
        required=False,
    )

    time_col = find_col(
        df,
        ["timestamp_sec", "time_sec", "timestamp", "frame_time", "seconds"],
        required=False,
    )

    base_candidates = [manifest_path.parent, BASE_DIR, PROJECT_DIR]

    rows = []

    for _, row in df.iterrows():
        frame_path = resolve_path(row[path_col], base_candidates)

        session_name = str(row[session_col]) if session_col else ""
        clip_name = str(row[clip_col]) if clip_col else ""

        if not clip_name or clip_name.lower() == "nan":
            if frame_path is not None:
                clip_name = frame_path.parent.name
            else:
                clip_name = "unknown_clip"

        if not session_name or session_name.lower() == "nan":
            if frame_path is not None and frame_path.parent.parent.name:
                session_name = frame_path.parent.parent.name
            else:
                session_name = "unknown_session"

        if "clip_group" in df.columns:
            clip_group = str(row["clip_group"])
        else:
            clip_group = f"{session_name}__{clip_name}"

        if frame_index_col:
            frame_index = int(safe_float(row[frame_index_col], default=0))
        else:
            frame_index = parse_frame_number(frame_path)

        timestamp = safe_float(row[time_col], default=np.nan) if time_col else np.nan

        rows.append({
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_name,
            "label": str(row[label_col]),
            "frame_index": frame_index,
            "timestamp_sec": timestamp,
            "frame_path": str(frame_path),
            "frame_exists": bool(frame_path is not None and frame_path.exists()),
        })

    out = pd.DataFrame(rows)
    out = out[out["frame_exists"]].copy()

    if out.empty:
        raise RuntimeError("No valid frame paths found in frame_manifest.csv")

    out = out.sort_values(["clip_group", "frame_index", "frame_path"]).reset_index(drop=True)

    return out, manifest_path


def get_bbox_columns(df):
    patterns = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
        ("bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"),
        ("box_x1", "box_y1", "box_x2", "box_y2"),
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for cols in patterns:
        if all(c in lower_map for c in cols):
            return tuple(lower_map[c] for c in cols), "xyxy"

    center_patterns = [
        ("x", "y", "width", "height"),
        ("center_x", "center_y", "width", "height"),
        ("cx", "cy", "w", "h"),
        ("bbox_center_x", "bbox_center_y", "bbox_width", "bbox_height"),
    ]

    for cols in center_patterns:
        if all(c in lower_map for c in cols):
            return tuple(lower_map[c] for c in cols), "xywh"

    return None, None


def load_detections(detections_path):
    if not detections_path:
        return {}

    detections_path = Path(detections_path)
    if not detections_path.is_absolute():
        detections_path = PROJECT_DIR / detections_path

    if not detections_path.exists():
        print(f"WARNING: detections file not found: {detections_path}")
        print("The script will run, but ROI-based drone features will be unavailable.")
        return {}

    df = pd.read_csv(detections_path)

    path_col = find_col(
        df,
        [
            "frame_path", "image_path", "saved_frame_path", "frame_file",
            "image_file", "filepath", "path"
        ],
        required=False,
    )

    conf_col = find_col(
        df,
        ["confidence", "conf", "score", "detection_confidence"],
        required=False,
    )

    bbox_cols, bbox_mode = get_bbox_columns(df)

    if path_col is None or bbox_cols is None:
        print("WARNING: Could not detect frame path or bbox columns in detections CSV.")
        print("Available columns:", list(df.columns))
        return {}

    base_candidates = [detections_path.parent, BASE_DIR, PROJECT_DIR]

    detections = {}

    for _, row in df.iterrows():
        frame_path = resolve_path(row[path_col], base_candidates)
        if frame_path is None:
            continue

        key = str(frame_path).lower().replace("\\", "/")

        conf = safe_float(row[conf_col], default=0.0) if conf_col else 0.0

        if bbox_mode == "xyxy":
            x1 = safe_float(row[bbox_cols[0]])
            y1 = safe_float(row[bbox_cols[1]])
            x2 = safe_float(row[bbox_cols[2]])
            y2 = safe_float(row[bbox_cols[3]])
        else:
            x = safe_float(row[bbox_cols[0]])
            y = safe_float(row[bbox_cols[1]])
            w = safe_float(row[bbox_cols[2]])
            h = safe_float(row[bbox_cols[3]])
            x1 = x - w / 2
            y1 = y - h / 2
            x2 = x + w / 2
            y2 = y + h / 2

        if not all(np.isfinite(v) for v in [x1, y1, x2, y2]):
            continue

        if x2 <= x1 or y2 <= y1:
            continue

        item = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "confidence": float(conf),
        }

        # If multiple detections exist for one frame, keep highest confidence.
        if key not in detections or item["confidence"] > detections[key]["confidence"]:
            detections[key] = item

    print(f"Loaded detections for {len(detections)} frames from {detections_path}")
    return detections


def read_image(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    return img


def resize_image_and_box(img, bbox, resize_width):
    h, w = img.shape[:2]

    if resize_width is None or resize_width <= 0 or w == resize_width:
        return img, bbox, 1.0, 1.0

    scale = resize_width / w
    new_w = resize_width
    new_h = int(round(h * scale))

    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    if bbox is None:
        return img_resized, None, scale, scale

    x1, y1, x2, y2 = bbox
    bbox_resized = [x1 * scale, y1 * scale, x2 * scale, y2 * scale]

    return img_resized, bbox_resized, scale, scale


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


def load_ptlflow_model(model_name, ckpt, device):
    print(f"Loading PTLFlow model: {model_name}")
    print(f"Checkpoint: {ckpt}")
    model = ptlflow.get_model(model_name, ckpt_path=ckpt)
    model = model.to(device)
    model.eval()
    return model


def compute_flow(img_a_bgr, img_b_bgr, model, device):
    images = [img_a_bgr, img_b_bgr]

    try:
        io_adapter = IOAdapter(model, img_a_bgr.shape[:2], cuda=(device.type == "cuda"))
    except TypeError:
        io_adapter = IOAdapter(model, img_a_bgr.shape[:2])

    inputs = io_adapter.prepare_inputs(images)
    inputs = move_to_device(inputs, device)

    with torch.inference_mode():
        predictions = model(inputs)

    if hasattr(io_adapter, "unpad_and_unscale"):
        try:
            predictions = io_adapter.unpad_and_unscale(predictions)
        except Exception:
            pass

    if "flows" not in predictions:
        raise RuntimeError(f"Model output missing 'flows'. Keys: {list(predictions.keys())}")

    flows = predictions["flows"]

    if flows.ndim == 5:
        flow = flows[0, 0]
    elif flows.ndim == 4:
        flow = flows[0]
    else:
        raise RuntimeError(f"Unexpected flow tensor shape: {flows.shape}")

    flow = flow.detach().cpu().float().numpy()
    flow = np.transpose(flow, (1, 2, 0))

    return flow.astype(np.float32)


def bbox_center(bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def make_roi(bbox_a, bbox_b, image_shape, margin_ratio=0.35):
    boxes = [b for b in [bbox_a, bbox_b] if b is not None]

    if not boxes:
        return None

    h, w = image_shape[:2]

    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)

    bw = x2 - x1
    bh = y2 - y1

    margin = max(bw, bh) * margin_ratio

    x1 = int(max(0, math.floor(x1 - margin)))
    y1 = int(max(0, math.floor(y1 - margin)))
    x2 = int(min(w, math.ceil(x2 + margin)))
    y2 = int(min(h, math.ceil(y2 + margin)))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def summarize_flow_in_roi(flow, roi, dt):
    h, w = flow.shape[:2]

    if roi is None:
        return {
            "flow_dx_norm_per_sec": 0.0,
            "flow_dy_norm_per_sec": 0.0,
            "flow_mag_norm_per_sec": 0.0,
            "flow_dx_mean_per_sec": 0.0,
            "flow_dy_mean_per_sec": 0.0,
            "flow_mag_mean_per_sec": 0.0,
            "flow_mag_mean": 0.0,
            "flow_mag_median": 0.0,
            "flow_mag_max": 0.0,
            "flow_mag_std": 0.0,
            "roi_width": 0.0,
            "roi_height": 0.0,
        }

    x1, y1, x2, y2 = roi
    roi_flow = flow[y1:y2, x1:x2, :]

    dx = roi_flow[:, :, 0]
    dy = roi_flow[:, :, 1]
    mag = np.sqrt(dx ** 2 + dy ** 2)

    finite = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(mag)

    if not finite.any():
        return {
            "flow_dx_norm_per_sec": 0.0,
            "flow_dy_norm_per_sec": 0.0,
            "flow_mag_norm_per_sec": 0.0,
            "flow_dx_mean_per_sec": 0.0,
            "flow_dy_mean_per_sec": 0.0,
            "flow_mag_mean_per_sec": 0.0,
            "flow_mag_mean": 0.0,
            "flow_mag_median": 0.0,
            "flow_mag_max": 0.0,
            "flow_mag_std": 0.0,
            "roi_width": float(x2 - x1),
            "roi_height": float(y2 - y1),
        }

    dxv = dx[finite]
    dyv = dy[finite]
    magv = mag[finite]

    dx_mean = float(dxv.mean())
    dy_mean = float(dyv.mean())
    mag_mean = float(magv.mean())

    return {
        "flow_dx_norm_per_sec": float((dx_mean / max(w, 1)) / dt),
        "flow_dy_norm_per_sec": float((dy_mean / max(h, 1)) / dt),
        "flow_mag_norm_per_sec": float((mag_mean / max(w, h, 1)) / dt),
        "flow_dx_mean_per_sec": float(dx_mean / dt),
        "flow_dy_mean_per_sec": float(dy_mean / dt),
        "flow_mag_mean_per_sec": float(mag_mean / dt),
        "flow_mag_mean": mag_mean,
        "flow_mag_median": float(np.median(magv)),
        "flow_mag_max": float(np.max(magv)),
        "flow_mag_std": float(np.std(magv)),
        "roi_width": float(x2 - x1),
        "roi_height": float(y2 - y1),
    }


def det_motion_features(bbox_a, bbox_b, conf_a, conf_b, image_shape, dt):
    h, w = image_shape[:2]

    center_a = bbox_center(bbox_a)
    center_b = bbox_center(bbox_b)

    if center_a is None or center_b is None:
        return {
            "det_dx": 0.0,
            "det_dy": 0.0,
            "det_speed": 0.0,
            "det_vx_norm_per_sec": 0.0,
            "det_vy_norm_per_sec": 0.0,
            "det_speed_norm_per_sec": 0.0,
            "conf_a": float(conf_a or 0.0),
            "conf_b": float(conf_b or 0.0),
        }

    dx = float(center_b[0] - center_a[0])
    dy = float(center_b[1] - center_a[1])
    speed = float(math.sqrt(dx * dx + dy * dy))

    return {
        "det_dx": dx,
        "det_dy": dy,
        "det_speed": speed,
        "det_vx_norm_per_sec": float((dx / max(w, 1)) / dt),
        "det_vy_norm_per_sec": float((dy / max(h, 1)) / dt),
        "det_speed_norm_per_sec": float((math.sqrt((dx / max(w, 1)) ** 2 + (dy / max(h, 1)) ** 2)) / dt),
        "conf_a": float(conf_a or 0.0),
        "conf_b": float(conf_b or 0.0),
    }


def flow_to_bgr(flow):
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]
    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    max_mag = np.percentile(mag[np.isfinite(mag)], 95) if np.isfinite(mag).any() else 1.0
    max_mag = max(float(max_mag), 1e-6)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = np.clip(ang * 180 / np.pi / 2, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_debug(img_a, img_b, flow, bbox_a, bbox_b, roi, out_path, title):
    a = img_a.copy()
    b = img_b.copy()
    flow_vis = flow_to_bgr(flow)

    def draw_box(img, bbox, color, label):
        if bbox is None:
            return

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    draw_box(a, bbox_a, (0, 255, 0), "current")
    draw_box(b, bbox_b, (0, 200, 255), "next")

    if roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(a, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.rectangle(flow_vis, (x1, y1), (x2, y2), (255, 255, 255), 2)

    ca = bbox_center(bbox_a)
    cb = bbox_center(bbox_b)

    if ca is not None and cb is not None:
        cv2.arrowedLine(
            a,
            (int(ca[0]), int(ca[1])),
            (int(cb[0]), int(cb[1])),
            (0, 0, 255),
            2,
            tipLength=0.3,
        )

    def add_title(img, text):
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(out, text, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        return out

    a = add_title(a, "Frame A + drone ROI")
    b = add_title(b, "Frame B")
    flow_vis = add_title(flow_vis, "DPFlow optical flow")

    target_w = 420

    def resize_panel(img):
        h, w = img.shape[:2]
        scale = target_w / w
        return cv2.resize(img, (target_w, int(h * scale)), interpolation=cv2.INTER_AREA)

    panels = [resize_panel(a), resize_panel(b), resize_panel(flow_vis)]

    hmax = max(p.shape[0] for p in panels)
    padded = []

    for p in panels:
        if p.shape[0] < hmax:
            pad = np.zeros((hmax - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        padded.append(p)

    combined = np.hstack(padded)

    cv2.rectangle(combined, (0, combined.shape[0] - 36), (combined.shape[1], combined.shape[0]), (0, 0, 0), -1)
    cv2.putText(combined, title[:150], (10, combined.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def build_clip_summary(seq_df):
    rows = []

    for clip_group, g in seq_df.groupby("clip_group"):
        g = g.sort_values("step_index").copy()

        label = g["label"].iloc[0]
        session_name = g["session_name"].iloc[0]
        clip_filename = g["clip_filename"].iloc[0]

        det_speed = g["det_speed_norm_per_sec"].fillna(0).to_numpy()
        dt_values = g["dt"].fillna(g["dt"].median() if len(g) else 0.1).to_numpy()

        accel_values = []
        for i in range(1, len(det_speed)):
            dt = max(float(dt_values[i]), 1e-6)
            accel_values.append(abs(float(det_speed[i] - det_speed[i - 1])) / dt)

        if not accel_values:
            accel_values = [0.0]

        row = {
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_filename,
            "label": label,
            "flow_method": g["flow_method"].iloc[0],
            "steps": int(len(g)),
            "roi_available_rate": float(g["roi_available"].mean()),
            "both_detected_rate": float(g["both_detected"].mean()),
            "any_detected_rate": float(g["any_detected"].mean()),

            "flow_mag_norm_per_sec_mean": float(g["flow_mag_norm_per_sec"].mean()),
            "flow_mag_norm_per_sec_max": float(g["flow_mag_norm_per_sec"].max()),
            "flow_mag_mean_mean": float(g["flow_mag_mean"].mean()),
            "flow_mag_max_max": float(g["flow_mag_max"].max()),

            "max_downward_flow": float(g["flow_dy_norm_per_sec"].max()),
            "max_upward_flow": float(abs(g["flow_dy_norm_per_sec"].min())),

            "det_speed_norm_per_sec_mean": float(g["det_speed_norm_per_sec"].mean()),
            "det_speed_norm_per_sec_max": float(g["det_speed_norm_per_sec"].max()),
            "max_downward_det_vy": float(g["det_vy_norm_per_sec"].max()),
            "max_upward_det_vy": float(abs(g["det_vy_norm_per_sec"].min())),

            "det_accel_mean": float(np.mean(accel_values)),
            "det_accel_max": float(np.max(accel_values)),

            "avg_conf_a": float(g["conf_a"].mean()),
            "avg_conf_b": float(g["conf_b"].mean()),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def write_registry_entry(run_summary, registry_path):
    registry_path.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "date": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_summary["run_name"],
        "stage": "feature_extraction",
        "dataset": "DroneAI current labeled clips",
        "flow_method": run_summary["flow_method"],
        "model": run_summary["model"],
        "checkpoint": run_summary["checkpoint"],
        "device": run_summary["device"],
        "total_clips": run_summary["total_clips"],
        "total_steps": run_summary["total_flow_steps"],
        "mean_roi_available_rate": run_summary["mean_roi_available_rate"],
        "mean_both_detected_rate": run_summary["mean_both_detected_rate"],
        "output_dir": run_summary["output_dir"],
        "notes": "DPFlow feature extraction run. Keep for paper tracking.",
    }

    new_df = pd.DataFrame([row])

    if registry_path.exists():
        old_df = pd.read_csv(registry_path)
        out_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_csv(registry_path, index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", default="LabelGUI/FrameDataset/frame_manifest.csv")
    parser.add_argument("--detections", default="LabelGUI/MotionResults/motion_v2_all/frame_detections.csv")

    parser.add_argument("--model", default="dpflow")
    parser.add_argument("--ckpt", default="things")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    parser.add_argument("--run-name", default="")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--resize-width", type=int, default=640)

    parser.add_argument("--max-clips", type=int, default=0)
    parser.add_argument("--max-pairs-per-clip", type=int, default=0)
    parser.add_argument("--debug-limit", type=int, default=50)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)

    run_name = args.run_name.strip()
    if not run_name:
        run_name = f"{args.model}_{args.ckpt}_droneai_{now_id()}"

    output_dir = BASE_DIR / "OpticalFlowResults" / run_name
    debug_dir = output_dir / "debug_flow_images"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== DroneAI DPFlow Feature Extraction ===")
    print("Run name:", run_name)
    print("Output:", output_dir)
    print("Device:", device)
    print("Model:", args.model)
    print("Checkpoint:", args.ckpt)
    print("Resize width:", args.resize_width)
    print("FPS fallback:", args.fps)

    manifest_df, manifest_path = load_manifest(args.manifest)
    detections = load_detections(args.detections)

    clip_groups = list(manifest_df["clip_group"].drop_duplicates())

    if args.max_clips and args.max_clips > 0:
        clip_groups = clip_groups[:args.max_clips]

    print(f"Loaded manifest frames: {len(manifest_df)}")
    print(f"Clips to process: {len(clip_groups)}")

    model = load_ptlflow_model(args.model, args.ckpt, device)

    sequence_rows = []
    debug_count = 0
    start_time = time.time()

    for clip_i, clip_group in enumerate(clip_groups):
        clip_df = manifest_df[manifest_df["clip_group"] == clip_group].copy()
        clip_df = clip_df.sort_values(["frame_index", "frame_path"]).reset_index(drop=True)

        if len(clip_df) < 2:
            continue

        label = clip_df["label"].iloc[0]
        print(f"\n[{clip_i + 1}/{len(clip_groups)}] {clip_group} | label={label} | frames={len(clip_df)}")

        pair_count = len(clip_df) - 1
        if args.max_pairs_per_clip and args.max_pairs_per_clip > 0:
            pair_count = min(pair_count, args.max_pairs_per_clip)

        for step_idx in range(pair_count):
            row_a = clip_df.iloc[step_idx]
            row_b = clip_df.iloc[step_idx + 1]

            path_a = Path(row_a["frame_path"])
            path_b = Path(row_b["frame_path"])

            key_a = str(path_a.resolve()).lower().replace("\\", "/")
            key_b = str(path_b.resolve()).lower().replace("\\", "/")

            det_a = detections.get(key_a)
            det_b = detections.get(key_b)

            bbox_a = det_a["bbox"] if det_a else None
            bbox_b = det_b["bbox"] if det_b else None
            conf_a = det_a["confidence"] if det_a else 0.0
            conf_b = det_b["confidence"] if det_b else 0.0

            try:
                img_a_original = read_image(path_a)
                img_b_original = read_image(path_b)

                img_a, bbox_a_scaled, _, _ = resize_image_and_box(img_a_original, bbox_a, args.resize_width)
                img_b, bbox_b_scaled, _, _ = resize_image_and_box(img_b_original, bbox_b, args.resize_width)

                if img_b.shape[:2] != img_a.shape[:2]:
                    img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_AREA)

                t_a = safe_float(row_a["timestamp_sec"], default=np.nan)
                t_b = safe_float(row_b["timestamp_sec"], default=np.nan)

                if np.isfinite(t_a) and np.isfinite(t_b) and t_b > t_a:
                    dt = float(t_b - t_a)
                else:
                    dt = 1.0 / max(args.fps, 1e-6)

                flow = compute_flow(img_a, img_b, model, device)

                roi = make_roi(bbox_a_scaled, bbox_b_scaled, img_a.shape, margin_ratio=0.35)

                roi_available = 1 if roi is not None else 0
                both_detected = 1 if bbox_a_scaled is not None and bbox_b_scaled is not None else 0
                any_detected = 1 if bbox_a_scaled is not None or bbox_b_scaled is not None else 0

                flow_feats = summarize_flow_in_roi(flow, roi, dt)
                det_feats = det_motion_features(
                    bbox_a_scaled,
                    bbox_b_scaled,
                    conf_a,
                    conf_b,
                    img_a.shape,
                    dt,
                )

                out_row = {
                    "clip_group": clip_group,
                    "session_name": row_a["session_name"],
                    "clip_filename": row_a["clip_filename"],
                    "label": label,
                    "flow_method": f"PTLFlow_{args.model}_{args.ckpt}",
                    "model": args.model,
                    "checkpoint": args.ckpt,
                    "step_index": int(step_idx),
                    "frame_a_index": int(row_a["frame_index"]),
                    "frame_b_index": int(row_b["frame_index"]),
                    "frame_a_path": str(path_a),
                    "frame_b_path": str(path_b),
                    "dt": float(dt),
                    "roi_available": roi_available,
                    "both_detected": both_detected,
                    "any_detected": any_detected,
                    **flow_feats,
                    **det_feats,
                }

                sequence_rows.append(out_row)

                if debug_count < args.debug_limit:
                    debug_name = f"{debug_count:04d}_{re.sub(r'[^A-Za-z0-9_.-]+', '_', clip_group)[:80]}_step{step_idx}.jpg"
                    debug_path = debug_dir / debug_name
                    draw_debug(
                        img_a,
                        img_b,
                        flow,
                        bbox_a_scaled,
                        bbox_b_scaled,
                        roi,
                        debug_path,
                        title=f"{clip_group} | {label} | step {step_idx}",
                    )
                    debug_count += 1

            except Exception as e:
                print(f"  ERROR step {step_idx}: {e}")
                sequence_rows.append({
                    "clip_group": clip_group,
                    "session_name": row_a["session_name"],
                    "clip_filename": row_a["clip_filename"],
                    "label": label,
                    "flow_method": f"PTLFlow_{args.model}_{args.ckpt}",
                    "model": args.model,
                    "checkpoint": args.ckpt,
                    "step_index": int(step_idx),
                    "frame_a_index": int(row_a["frame_index"]),
                    "frame_b_index": int(row_b["frame_index"]),
                    "frame_a_path": str(path_a),
                    "frame_b_path": str(path_b),
                    "dt": 1.0 / max(args.fps, 1e-6),
                    "roi_available": 0,
                    "both_detected": 0,
                    "any_detected": 0,
                    "error": str(e),
                })

    seq_df = pd.DataFrame(sequence_rows)

    sequence_csv = output_dir / "flow_sequence_features.csv"
    seq_df.to_csv(sequence_csv, index=False)

    valid_seq_df = seq_df.copy()

    for col in [
        "roi_available", "both_detected", "any_detected",
        "flow_mag_norm_per_sec", "flow_dy_norm_per_sec",
        "det_speed_norm_per_sec", "det_vy_norm_per_sec",
        "conf_a", "conf_b",
    ]:
        if col not in valid_seq_df.columns:
            valid_seq_df[col] = 0.0

    clip_summary_df = build_clip_summary(valid_seq_df)
    clip_summary_csv = output_dir / "clip_flow_summary.csv"
    clip_summary_df.to_csv(clip_summary_csv, index=False)

    elapsed = time.time() - start_time

    run_summary = {
        "run_name": run_name,
        "flow_method": f"PTLFlow_{args.model}_{args.ckpt}",
        "model": args.model,
        "checkpoint": args.ckpt,
        "device": str(device),
        "manifest": str(manifest_path),
        "detections": str(args.detections),
        "resize_width": args.resize_width,
        "fps_fallback": args.fps,
        "total_clips": int(clip_summary_df["clip_group"].nunique()) if len(clip_summary_df) else 0,
        "total_flow_steps": int(len(seq_df)),
        "debug_images_saved": int(debug_count),
        "mean_roi_available_rate": float(clip_summary_df["roi_available_rate"].mean()) if len(clip_summary_df) else None,
        "mean_both_detected_rate": float(clip_summary_df["both_detected_rate"].mean()) if len(clip_summary_df) else None,
        "elapsed_seconds": float(elapsed),
        "output_dir": str(output_dir),
        "sequence_features_csv": str(sequence_csv),
        "clip_summary_csv": str(clip_summary_csv),
        "debug_images": str(debug_dir),
    }

    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "script": "extract_dpflow_drone_features.py",
            "purpose": "Extract DPFlow optical-flow features from DroneAI labeled clips.",
            "model": args.model,
            "checkpoint": args.ckpt,
            "device": str(device),
            "resize_width": args.resize_width,
            "fps_fallback": args.fps,
            "notes": "This run should be preserved for experiment tracking and paper writing.",
        }, f, indent=2)

    with open(output_dir / "notes.txt", "w", encoding="utf-8") as f:
        f.write(
            "DroneAI DPFlow feature extraction run.\n"
            "This run uses PTLFlow DPFlow optical flow to generate sequence-level motion features.\n"
            "Do not delete this folder. Results may be used for paper tracking.\n"
        )

    registry_path = BASE_DIR / "TrainingRuns" / "experiment_registry.csv"
    write_registry_entry(run_summary, registry_path)

    print("\n=== Done ===")
    print(json.dumps(run_summary, indent=2))
    print("\nSaved:")
    print("Sequence features:", sequence_csv)
    print("Clip summary:", clip_summary_csv)
    print("Run summary:", output_dir / "run_summary.json")
    print("Debug images:", debug_dir)
    print("Experiment registry:", registry_path)


if __name__ == "__main__":
    main()
