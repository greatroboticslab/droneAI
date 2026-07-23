import json
from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

OPTICAL_FLOW_DIR = BASE_DIR / "OpticalFlowResults" / "flow_v1"
OPTICAL_FLOW_RUN_SUMMARY = OPTICAL_FLOW_DIR / "run_summary.json"
OPTICAL_FLOW_CLIP_SUMMARY = OPTICAL_FLOW_DIR / "clip_flow_summary.csv"
OPTICAL_FLOW_SEQUENCE_FEATURES = OPTICAL_FLOW_DIR / "flow_sequence_features.csv"
OPTICAL_FLOW_DEBUG_DIR = OPTICAL_FLOW_DIR / "debug_flow_images"


LABEL_ORDER = ["takeoff", "land", "minor-crash", "severe-crash"]


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_float(value, digits=4):
    return f"{safe_float(value):.{digits}f}"


def format_percent(value):
    return f"{safe_float(value) * 100:.2f}%"


def read_json(path: Path):
    if not path.exists():
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def safe_clip_name(clip_group):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(clip_group))


def ordered_labels(labels):
    labels = list(labels)

    ordered = []

    for label in LABEL_ORDER:
        if label in labels:
            ordered.append(label)

    for label in sorted(labels):
        if label not in ordered:
            ordered.append(label)

    return ordered


def mean_column(group, col):
    if col not in group.columns:
        return 0.0

    values = pd.to_numeric(group[col], errors="coerce").fillna(0.0)
    return float(values.mean()) if len(values) else 0.0


def load_clip_summary():
    if not OPTICAL_FLOW_CLIP_SUMMARY.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(OPTICAL_FLOW_CLIP_SUMMARY)
    except Exception:
        return pd.DataFrame()

    if "label" not in df.columns:
        return pd.DataFrame()

    df["label"] = df["label"].astype(str)

    return df


def load_flow_sequence_features():
    if not OPTICAL_FLOW_SEQUENCE_FEATURES.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(OPTICAL_FLOW_SEQUENCE_FEATURES)
    except Exception:
        return pd.DataFrame()

    required = {"clip_group", "label", "step_index"}

    if not required.issubset(df.columns):
        return pd.DataFrame()

    df["clip_group"] = df["clip_group"].astype(str)
    df["label"] = df["label"].astype(str)
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce").fillna(0).astype(int)

    return df


def build_class_stats(df):
    if df.empty:
        return []

    labels = ordered_labels(df["label"].unique())

    rows = []

    for label in labels:
        group = df[df["label"] == label].copy()

        row = {
            "label": label,
            "clips": int(len(group)),

            "roi_available_rate": mean_column(group, "roi_available_rate"),
            "both_detected_rate": mean_column(group, "both_detected_rate"),

            "avg_flow_speed": mean_column(group, "flow_mag_norm_per_sec_mean"),
            "peak_flow_speed": mean_column(group, "flow_mag_norm_per_sec_max"),
            "peak_downward_flow": mean_column(group, "max_downward_flow"),
            "peak_upward_flow": mean_column(group, "max_upward_flow"),

            "avg_detected_speed": mean_column(group, "det_speed_norm_per_sec_mean"),
            "peak_detected_speed": mean_column(group, "det_speed_norm_per_sec_max"),
            "peak_downward_det_vy": mean_column(group, "max_downward_det_vy"),
            "peak_upward_det_vy": mean_column(group, "max_upward_det_vy"),

            "avg_detected_accel": mean_column(group, "det_accel_mean"),
            "peak_detected_accel": mean_column(group, "det_accel_max"),
        }

        rows.append(row)

    return rows


def make_bar_chart(title, rows, value_key, display_digits=4):
    max_value = max([safe_float(r.get(value_key, 0.0)) for r in rows], default=0.0)

    bars = []

    for row in rows:
        value = safe_float(row.get(value_key, 0.0))
        width = 0.0 if max_value <= 0 else (value / max_value) * 100.0

        bars.append({
            "label": row["label"],
            "value": value,
            "display": format_float(value, display_digits),
            "width": round(width, 2),
        })

    return {
        "title": title,
        "value_key": value_key,
        "bars": bars,
    }


def list_debug_images(limit=12):
    if not OPTICAL_FLOW_DEBUG_DIR.exists():
        return []

    image_files = []

    for path in sorted(OPTICAL_FLOW_DEBUG_DIR.iterdir()):
        if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            image_files.append(path.name)

    return image_files[:limit]


def list_debug_images_for_clip(clip_group, limit=12):
    if not OPTICAL_FLOW_DEBUG_DIR.exists():
        return []

    safe_clip = safe_clip_name(clip_group)
    matched = []

    for path in sorted(OPTICAL_FLOW_DEBUG_DIR.iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        if safe_clip in path.name:
            matched.append(path.name)

    return matched[:limit]


def load_optical_flow_dashboard_data():
    run_summary = read_json(OPTICAL_FLOW_RUN_SUMMARY)
    clip_df = load_clip_summary()

    available = bool(run_summary) and not clip_df.empty

    if not available:
        return {
            "available": False,
            "message": (
                "Optical-flow results were not found. Expected files: "
                f"{OPTICAL_FLOW_RUN_SUMMARY} and {OPTICAL_FLOW_CLIP_SUMMARY}"
            ),
        }

    class_stats = build_class_stats(clip_df)

    charts = [
        make_bar_chart("Average Optical-Flow Speed by Class", class_stats, "avg_flow_speed"),
        make_bar_chart("Average Detected-Center Speed by Class", class_stats, "avg_detected_speed"),
        make_bar_chart("Peak Detected-Center Speed by Class", class_stats, "peak_detected_speed"),
        make_bar_chart("Average Detected Acceleration by Class", class_stats, "avg_detected_accel"),
        make_bar_chart("Peak Downward Flow by Class", class_stats, "peak_downward_flow"),
        make_bar_chart("Both-Frame Detection Rate by Class", class_stats, "both_detected_rate"),
    ]

    cards = [
        {
            "title": "Total Clips",
            "value": str(run_summary.get("total_clips", clip_df["clip_group"].nunique() if "clip_group" in clip_df.columns else len(clip_df))),
        },
        {
            "title": "Optical-Flow Steps",
            "value": str(run_summary.get("total_flow_steps", "N/A")),
        },
        {
            "title": "ROI Available Rate",
            "value": format_percent(run_summary.get("mean_roi_available_rate", 0.0)),
        },
        {
            "title": "Both-Detected Rate",
            "value": format_percent(run_summary.get("mean_both_detected_rate", 0.0)),
        },
    ]

    return {
        "available": True,
        "run_summary": run_summary,
        "cards": cards,
        "class_stats": class_stats,
        "charts": charts,
        "debug_images": list_debug_images(limit=12),
        "files": {
            "run_summary": str(OPTICAL_FLOW_RUN_SUMMARY),
            "clip_summary": str(OPTICAL_FLOW_CLIP_SUMMARY),
            "sequence_features": str(OPTICAL_FLOW_SEQUENCE_FEATURES),
            "debug_dir": str(OPTICAL_FLOW_DEBUG_DIR),
        },
    }


def numeric_series(group, col):
    if col not in group.columns:
        return []

    return pd.to_numeric(group[col], errors="coerce").fillna(0.0).astype(float).tolist()


def make_line_chart(title, values, subtitle="", width=700, height=220):
    values = [safe_float(v) for v in values]

    if not values:
        return {
            "title": title,
            "subtitle": subtitle,
            "available": False,
            "points": "",
            "min_value": 0.0,
            "max_value": 0.0,
            "mean_value": 0.0,
        }

    min_v = min(values)
    max_v = max(values)
    mean_v = sum(values) / len(values)

    if abs(max_v - min_v) < 1e-9:
        y_values = [height / 2 for _ in values]
    else:
        y_values = [
            height - ((v - min_v) / (max_v - min_v)) * (height - 30) - 15
            for v in values
        ]

    if len(values) == 1:
        x_values = [width / 2]
    else:
        x_values = [
            (i / (len(values) - 1)) * (width - 40) + 20
            for i in range(len(values))
        ]

    points = " ".join([f"{x:.2f},{y:.2f}" for x, y in zip(x_values, y_values)])

    zero_line_y = None
    if min_v < 0 < max_v:
        zero_line_y = height - ((0 - min_v) / (max_v - min_v)) * (height - 30) - 15

    return {
        "title": title,
        "subtitle": subtitle,
        "available": True,
        "points": points,
        "min_value": min_v,
        "max_value": max_v,
        "mean_value": mean_v,
        "zero_line_y": zero_line_y,
        "width": width,
        "height": height,
    }


def make_trajectory_chart(group, width=420, height=320):
    dx = numeric_series(group, "det_dx")
    dy = numeric_series(group, "det_dy")

    if not dx or not dy:
        return {
            "available": False,
            "points": "",
            "width": width,
            "height": height,
        }

    xs = [0.0]
    ys = [0.0]

    for i in range(min(len(dx), len(dy))):
        xs.append(xs[-1] + safe_float(dx[i]))
        ys.append(ys[-1] + safe_float(dy[i]))

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    if abs(max_x - min_x) < 1e-9:
        plot_x = [width / 2 for _ in xs]
    else:
        plot_x = [
            ((x - min_x) / (max_x - min_x)) * (width - 40) + 20
            for x in xs
        ]

    if abs(max_y - min_y) < 1e-9:
        plot_y = [height / 2 for _ in ys]
    else:
        plot_y = [
            ((y - min_y) / (max_y - min_y)) * (height - 40) + 20
            for y in ys
        ]

    points = " ".join([f"{x:.2f},{y:.2f}" for x, y in zip(plot_x, plot_y)])

    return {
        "available": True,
        "points": points,
        "width": width,
        "height": height,
        "start_x": plot_x[0],
        "start_y": plot_y[0],
        "end_x": plot_x[-1],
        "end_y": plot_y[-1],
        "raw_dx_total": xs[-1],
        "raw_dy_total": ys[-1],
    }


def make_acceleration(values):
    values = [safe_float(v) for v in values]

    if len(values) < 2:
        return [0.0 for _ in values]

    accel = [0.0]

    for i in range(1, len(values)):
        accel.append(abs(values[i] - values[i - 1]))

    return accel


def make_clip_list(clip_df):
    if clip_df.empty:
        return []

    rows = []

    sort_cols = []
    for col in ["label", "clip_filename", "clip_group"]:
        if col in clip_df.columns:
            sort_cols.append(col)

    if sort_cols:
        clip_df = clip_df.sort_values(sort_cols)

    for _, row in clip_df.iterrows():
        rows.append({
            "clip_group": str(row.get("clip_group", "")),
            "label": str(row.get("label", "")),
            "session_name": str(row.get("session_name", "")),
            "clip_filename": str(row.get("clip_filename", "")),
            "steps": int(safe_float(row.get("steps", 0))),
            "roi_available_rate": safe_float(row.get("roi_available_rate", 0.0)),
            "both_detected_rate": safe_float(row.get("both_detected_rate", 0.0)),
            "avg_detected_speed": safe_float(row.get("det_speed_norm_per_sec_mean", 0.0)),
            "peak_detected_speed": safe_float(row.get("det_speed_norm_per_sec_max", 0.0)),
            "avg_flow_speed": safe_float(row.get("flow_mag_norm_per_sec_mean", 0.0)),
            "avg_accel": safe_float(row.get("det_accel_mean", 0.0)),
        })

    return rows


def load_optical_flow_clip_explorer_data(selected_clip_group=""):
    clip_df = load_clip_summary()
    seq_df = load_flow_sequence_features()

    available = not clip_df.empty and not seq_df.empty

    if not available:
        return {
            "available": False,
            "message": (
                "Clip Explorer needs both clip_flow_summary.csv and flow_sequence_features.csv. "
                f"Expected files: {OPTICAL_FLOW_CLIP_SUMMARY} and {OPTICAL_FLOW_SEQUENCE_FEATURES}"
            ),
        }

    clip_list = make_clip_list(clip_df)

    if not selected_clip_group and clip_list:
        selected_clip_group = clip_list[0]["clip_group"]

    selected_summary = clip_df[clip_df["clip_group"].astype(str) == str(selected_clip_group)]
    selected_steps = seq_df[seq_df["clip_group"].astype(str) == str(selected_clip_group)].copy()

    if selected_summary.empty or selected_steps.empty:
        selected_clip_group = clip_list[0]["clip_group"]
        selected_summary = clip_df[clip_df["clip_group"].astype(str) == str(selected_clip_group)]
        selected_steps = seq_df[seq_df["clip_group"].astype(str) == str(selected_clip_group)].copy()

    selected_steps = selected_steps.sort_values("step_index")

    summary_row = selected_summary.iloc[0].to_dict()

    speed = numeric_series(selected_steps, "det_speed_norm_per_sec")
    vertical_velocity = numeric_series(selected_steps, "det_vy_norm_per_sec")
    horizontal_velocity = numeric_series(selected_steps, "det_vx_norm_per_sec")
    flow_magnitude = numeric_series(selected_steps, "flow_mag_norm_per_sec")
    acceleration = make_acceleration(speed)

    charts = [
        make_line_chart(
            "Detected-Center Speed Over Time",
            speed,
            "Normalized movement of the detected drone center between frames.",
        ),
        make_line_chart(
            "Vertical Velocity Over Time",
            vertical_velocity,
            "Positive values mean downward motion in image coordinates. Negative values mean upward motion.",
        ),
        make_line_chart(
            "Acceleration Over Time",
            acceleration,
            "Change in detected-center speed between consecutive optical-flow steps.",
        ),
        make_line_chart(
            "Optical-Flow Magnitude Over Time",
            flow_magnitude,
            "Average normalized optical-flow motion inside the drone-focused region.",
        ),
        make_line_chart(
            "Horizontal Velocity Over Time",
            horizontal_velocity,
            "Left/right movement of the detected drone center.",
        ),
    ]

    trajectory = make_trajectory_chart(selected_steps)

    step_rows = []
    for _, row in selected_steps.iterrows():
        step_rows.append({
            "step_index": int(safe_float(row.get("step_index", 0))),
            "roi_available": str(row.get("roi_available", "")),
            "both_detected": str(row.get("both_detected", "")),
            "det_speed_norm_per_sec": safe_float(row.get("det_speed_norm_per_sec", 0.0)),
            "det_vx_norm_per_sec": safe_float(row.get("det_vx_norm_per_sec", 0.0)),
            "det_vy_norm_per_sec": safe_float(row.get("det_vy_norm_per_sec", 0.0)),
            "flow_mag_norm_per_sec": safe_float(row.get("flow_mag_norm_per_sec", 0.0)),
            "flow_dy_norm_per_sec": safe_float(row.get("flow_dy_norm_per_sec", 0.0)),
            "conf_a": safe_float(row.get("conf_a", 0.0)),
            "conf_b": safe_float(row.get("conf_b", 0.0)),
        })

    selected = {
        "clip_group": selected_clip_group,
        "label": str(summary_row.get("label", "")),
        "session_name": str(summary_row.get("session_name", "")),
        "clip_filename": str(summary_row.get("clip_filename", "")),
        "steps": int(safe_float(summary_row.get("steps", len(selected_steps)))),
        "roi_available_rate": safe_float(summary_row.get("roi_available_rate", 0.0)),
        "both_detected_rate": safe_float(summary_row.get("both_detected_rate", 0.0)),
        "avg_detected_speed": safe_float(summary_row.get("det_speed_norm_per_sec_mean", 0.0)),
        "peak_detected_speed": safe_float(summary_row.get("det_speed_norm_per_sec_max", 0.0)),
        "avg_flow_speed": safe_float(summary_row.get("flow_mag_norm_per_sec_mean", 0.0)),
        "peak_flow_speed": safe_float(summary_row.get("flow_mag_norm_per_sec_max", 0.0)),
        "avg_accel": safe_float(summary_row.get("det_accel_mean", 0.0)),
        "peak_accel": safe_float(summary_row.get("det_accel_max", 0.0)),
        "max_downward_flow": safe_float(summary_row.get("max_downward_flow", 0.0)),
        "max_upward_flow": safe_float(summary_row.get("max_upward_flow", 0.0)),
    }

    return {
        "available": True,
        "clip_list": clip_list,
        "selected": selected,
        "charts": charts,
        "trajectory": trajectory,
        "step_rows": step_rows,
        "debug_images": list_debug_images_for_clip(selected_clip_group, limit=12),
        "files": {
            "clip_summary": str(OPTICAL_FLOW_CLIP_SUMMARY),
            "sequence_features": str(OPTICAL_FLOW_SEQUENCE_FEATURES),
            "debug_dir": str(OPTICAL_FLOW_DEBUG_DIR),
        },
    }
