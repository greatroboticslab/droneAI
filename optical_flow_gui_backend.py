import json
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

OPTICAL_FLOW_DIR = BASE_DIR / "OpticalFlowResults" / "flow_v1"
OPTICAL_FLOW_RUN_SUMMARY = OPTICAL_FLOW_DIR / "run_summary.json"
OPTICAL_FLOW_CLIP_SUMMARY = OPTICAL_FLOW_DIR / "clip_flow_summary.csv"
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

            # Coverage / quality
            "roi_available_rate": mean_column(group, "roi_available_rate"),
            "both_detected_rate": mean_column(group, "both_detected_rate"),

            # Optical-flow motion features
            "avg_flow_speed": mean_column(group, "flow_mag_norm_per_sec_mean"),
            "peak_flow_speed": mean_column(group, "flow_mag_norm_per_sec_max"),
            "peak_downward_flow": mean_column(group, "max_downward_flow"),
            "peak_upward_flow": mean_column(group, "max_upward_flow"),

            # Detector-center motion features
            "avg_detected_speed": mean_column(group, "det_speed_norm_per_sec_mean"),
            "peak_detected_speed": mean_column(group, "det_speed_norm_per_sec_max"),
            "peak_downward_det_vy": mean_column(group, "max_downward_det_vy"),
            "peak_upward_det_vy": mean_column(group, "max_upward_det_vy"),

            # Acceleration
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
            "debug_dir": str(OPTICAL_FLOW_DEBUG_DIR),
        },
    }
