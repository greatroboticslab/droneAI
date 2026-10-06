"""
Data for the optical-flow training dashboard (/optical_flow).

Reads what the ML scripts already write, nothing else:
    LabelGUI/TrainingRuns/<run>/metrics.json, predictions.csv, feature_importance.csv
    LabelGUI/OpticalFlowResults/<features run>/run_summary.json
and, read-only, the labeling DB for how many events are labeled so far.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from data_snapshot import SNAPSHOTS_DIR, list_snapshots
from repo_paths import REPO_ROOT, resolve_path

BASE_DIR = Path(__file__).resolve().parent
TRAINING_RUNS_DIR = BASE_DIR / "TrainingRuns"

EVENTS = ["takeoff", "land", "minor-crash", "severe-crash"]
EVENT_NAMES = {"takeoff": "Take-off", "land": "Land",
               "minor-crash": "Minor crash", "severe-crash": "Severe crash"}
MODEL_NAMES = {"random_forest": "Random forest", "extra_trees": "Extra trees",
               "gradient_boosting": "Gradient boosting",
               "hist_gradient_boosting": "Histogram gradient boosting",
               "xgboost": "XGBoost", "svm_rbf": "SVM (RBF kernel)",
               "logistic_regression": "Logistic regression"}
EVENTS_PER_CLASS_TARGET = 50   # Gate 1 in ROADMAP.md
ACCURACY_TARGET = 0.85         # Gate 2 in ROADMAP.md
HONEST_SPLIT = "grouped_kfold_video"

# Plain names for feature columns: longest matching prefix, then a suffix.
_SIGNALS = [
    ("frame_mag_std_norm_per_sec", "Speed spread across the frame"),
    ("frame_mag_p90_norm_per_sec", "Speed of the fastest parts"),
    ("frame_mag_norm_per_sec", "Whole-frame speed"),
    ("frame_mag_accel", "Change in whole-frame speed"),
    ("frame_dx_norm_per_sec", "Sideways motion"),
    ("frame_dy_norm_per_sec", "Up/down motion"),
    ("frame_divergence_per_sec", "Zoom (approach or drop)"),
    ("frame_curl_per_sec", "Spin"),
    ("frame_still_fraction", "Still part of the frame"),
    ("frame_coherence", "How uniform the motion is"),
    ("flow_mag", "Drone-box speed"),
    ("flow_dx", "Drone-box sideways motion"),
    ("flow_dy", "Drone-box up/down motion"),
    ("det_vy", "Drone box up/down speed"),
    ("det_vx", "Drone box sideways speed"),
    ("det_speed", "Drone box speed"),
    ("det_dy", "Drone box up/down move"),
    ("det_dx", "Drone box sideways move"),
    ("det_max_downward_vy", "Drone box fastest drop"),
    ("det_max_upward_vy", "Drone box fastest rise"),
    ("det_down_up_ratio", "Drone box drop vs rise"),
    ("det_vertical_horizontal_ratio", "Drone box vertical vs sideways"),
    ("flow_max_downward_dy", "Drone-box fastest downward flow"),
    ("flow_max_upward_dy", "Drone-box fastest upward flow"),
    ("flow_down_up_ratio", "Drone-box downward vs upward flow"),
    ("flow_vertical_horizontal_ratio", "Drone-box vertical vs sideways flow"),
    ("conf_", "Detector confidence"),
    ("det_", "Drone box"),
    ("roi_", "Drone box size"),
    ("duration_estimate_sec", "Clip length"),
    ("num_steps", "Frames in clip"),
]
_SUFFIXES = [
    ("_end_minus_start", "end minus start"), ("_start_mean", "at the start"),
    ("_mid_mean", "in the middle"), ("_end_mean", "at the end"),
    ("_slope_first_last", "trend"), ("_peak_to_mean", "spikiness"),
    ("_jerk", "jerkiness"), ("_first", "first frame"), ("_last", "last frame"),
    ("_delta", "change"), ("_mean", "average"), ("_median", "median"), ("_std", "variation"),
    ("_max", "maximum"), ("_min", "minimum"), ("_p90", "90th percentile"),
    ("_p75", "75th percentile"), ("_p25", "25th percentile"), ("_p10", "10th percentile"),
]


def feature_label(column):
    matches = [(p, label) for p, label in _SIGNALS if column.startswith(p)]
    if not matches:
        return column
    prefix, name = max(matches, key=lambda m: len(m[0]))
    rest = column[len(prefix):]
    part = next((label for suffix, label in _SUFFIXES if rest.endswith(suffix)), "")
    if rest.startswith("_accel") and "Change" not in name:
        part = ("change, " + part).strip(", ")
    return f"{name}, {part}" if part else name


def _read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _read_csv(path):
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _run_date(run_dir, metrics=None):
    # Recorded by the trainer; file times change when a run is copied between machines.
    try:
        return datetime.fromisoformat((metrics or {})["created_at"])
    except (KeyError, TypeError, ValueError):
        pass
    try:
        return datetime.fromtimestamp((run_dir / "metrics.json").stat().st_mtime)
    except OSError:
        return None


def list_training_runs(runs_dir=TRAINING_RUNS_DIR):
    """Every tabular training run, newest first."""
    runs = []
    for metrics_path in Path(runs_dir).glob("*/metrics.json"):
        m = _read_json(metrics_path)
        if not m or "accuracy" not in m:
            continue
        run_dir = metrics_path.parent
        honest = m.get("split_mode") == HONEST_SPLIT
        date = _run_date(run_dir, m)
        runs.append({
            "name": run_dir.name,
            "date": date,
            # Formatted here: "%-d" is not available on Windows.
            "date_label": f"{date.day} {date:%b %Y, %H:%M}" if date else "",
            "date_short": f"{date.day} {date:%b %Y}" if date else "-",
            "honest": honest,
            "split": (f"{m.get('folds', 5)}-fold, grouped by video" if honest
                      else f"single {m.get('split_mode', '?')} split"),
            "view": m.get("view_type_filter") or "all",
            "clips": m.get("total_clips"),
            "videos": m.get("total_videos"),
            "accuracy": m.get("accuracy"),
            "accuracy_std": m.get("accuracy_std"),
            "macro_f1": m.get("macro_f1"),
            "model": m.get("best_model", ""),
            "snapshot": m.get("data_snapshot") or None,
        })
    runs.sort(key=lambda r: r["date"] or datetime.min, reverse=True)
    return runs


def _per_class(pred):
    rows = []
    for ev in EVENTS:
        truth = pred["true_label"] == ev
        guessed = pred["pred_label"] == ev
        hits = int((truth & guessed).sum())
        n, g = int(truth.sum()), int(guessed.sum())
        if not n and not g:
            continue
        recall = hits / n if n else 0.0
        precision = hits / g if g else 0.0
        f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
        rows.append({"event": ev, "name": EVENT_NAMES[ev], "clips": n, "found": hits,
                     "predicted": g, "recall": recall, "precision": precision, "f1": f1})
    return rows


def _confusion(pred, labels):
    grid = []
    for t in labels:
        row = pred[pred["true_label"] == t]
        total = len(row)
        cells = []
        for p in labels:
            n = int((row["pred_label"] == p).sum())
            cells.append({"n": n, "share": n / total if total else 0.0, "diagonal": t == p})
        grid.append({"label": EVENT_NAMES.get(t, t), "total": total, "cells": cells})
    return grid


def _mistakes(pred, limit=8):
    wrong = pred[pred["true_label"] != pred["pred_label"]].copy()
    if wrong.empty:
        return []
    prob_cols = {c[5:]: c for c in pred.columns if c.startswith("prob_")}
    wrong["confidence"] = [row.get(prob_cols.get(row["pred_label"], ""), float("nan"))
                           for _, row in wrong.iterrows()]
    wrong = wrong.sort_values("confidence", ascending=False).head(limit)
    return [{
        "clip": str(r.get("clip_group", "")),
        "video_id": str(r.get("video_id", "")),
        "truth": EVENT_NAMES.get(r["true_label"], r["true_label"]),
        "guess": EVENT_NAMES.get(r["pred_label"], r["pred_label"]),
        "confidence": None if pd.isna(r["confidence"]) else float(r["confidence"]),
    } for _, r in wrong.iterrows()]


def _extraction_info(features_csv):
    if not features_csv:
        return {}
    path = resolve_path(features_csv)
    summary = _read_json(path.parent / "run_summary.json") if path else {}
    if not summary:
        return {}
    seconds = summary.get("elapsed_seconds")
    return {
        "name": summary.get("run_name", path.parent.name),
        "flow_model": f"{summary.get('model', '?')} ({summary.get('checkpoint', '?')} weights)",
        "device": {"mps": "Apple GPU (MPS)", "cuda": "NVIDIA GPU (CUDA)",
                   "cpu": "CPU"}.get(summary.get("device"), summary.get("device", "?")),
        "clips": summary.get("total_clips"),
        "steps": summary.get("total_flow_steps"),
        "resize": summary.get("resize_width"),
        "minutes": round(seconds / 60, 1) if seconds else None,
        "drone_box_rate": summary.get("mean_roi_available_rate"),
    }


def labeled_event_counts(db_path):
    """Events per class in the active dataset's labeled videos (DB opened read-only)."""
    try:
        from video_manifest import build_video_manifest
        rows = build_video_manifest(db_path, active_only=True)
    except Exception:
        return None
    labeled = [r for r in rows if r.get("status") == "labeled"]
    out = {"videos": len(labeled), "classes": []}
    for ev in EVENTS:
        col = "n_" + ev.replace("-", "_")
        total = sum(int(r.get(col) or 0) for r in labeled)
        fpv = sum(int(r.get(col) or 0) for r in labeled if r.get("view_type") == "FPV")
        out["classes"].append({"name": EVENT_NAMES[ev], "total": total, "fpv": fpv,
                               "share": min(total / EVENTS_PER_CLASS_TARGET, 1.0)})
    return out


def load_training_dashboard(run_name="", runs_dir=TRAINING_RUNS_DIR, db_path=None,
                            snapshots_dir=SNAPSHOTS_DIR):
    runs = list_training_runs(runs_dir)
    data = {"runs": runs, "run": None, "accuracy_target": ACCURACY_TARGET,
            "snapshots": list_snapshots(snapshots_dir),
            "events_target": EVENTS_PER_CLASS_TARGET,
            "labeled": labeled_event_counts(db_path) if db_path else None}
    if not runs:
        return data

    chosen = next((r for r in runs if r["name"] == run_name), None)
    if chosen is None:
        chosen = next((r for r in runs if r["honest"]), runs[0])
    run_dir = Path(runs_dir) / chosen["name"]
    m = _read_json(run_dir / "metrics.json")
    pred = _read_csv(run_dir / "predictions.csv")
    has_pred = {"true_label", "pred_label"} <= set(pred.columns)

    counts = m.get("label_counts") or {}
    total = sum(counts.values()) or 0
    majority = max(counts.values()) / total if total else None
    labels = [e for e in EVENTS if e in counts] or EVENTS

    importance = _read_csv(run_dir / "feature_importance.csv")
    top_features = []
    if {"feature", "importance"} <= set(importance.columns):
        importance = importance.sort_values("importance", ascending=False).head(10)
        peak = float(importance["importance"].max() or 1.0)
        top_features = [{"label": feature_label(r["feature"]), "column": r["feature"],
                         "share": float(r["importance"]) / peak}
                        for _, r in importance.iterrows()]

    folds = [{"fold": int(f.get("fold", i)) + 1, "accuracy": f.get("accuracy"),
              "clips": f.get("val_clips")} for i, f in enumerate(m.get("per_fold_best_model") or [])]

    features_csv = m.get("input_features_csv", "")
    command = ["python LabelGUI/train_dpflow_tabular_models.py"]
    if features_csv:
        command.append(f"--features-csv {features_csv}")
    if m.get("view_type_filter") and m.get("view_type_filter") != "all":
        command.append(f"--view-type {m['view_type_filter']}")
    command.append(f"--run-name {chosen['name']}")

    data["run"] = {
        **chosen,
        "macro_f1_std": m.get("macro_f1_std"),
        "majority": majority,
        "majority_label": EVENT_NAMES.get(max(counts, key=counts.get), "") if counts else "",
        "chance": 1 / len(labels) if labels else None,
        "label_counts": [{"name": EVENT_NAMES.get(e, e), "n": counts.get(e, 0)} for e in labels],
        "held_out_videos": m.get("held_out_test_videos", 0),
        "held_out_clips": m.get("held_out_clips_removed", 0),
        "clips_before": m.get("clips_before_filters"),
        "feature_count": m.get("feature_count"),
        "seed": m.get("seed"),
        "features_csv": features_csv,
        "extraction": _extraction_info(features_csv),
        "folds": folds,
        "model_name": MODEL_NAMES.get(chosen["model"], chosen["model"]),
        "models": sorted(({"name": MODEL_NAMES.get(r.get("model"), r.get("model", "?")),
                           "accuracy": r.get("accuracy") or 0.0,
                           "accuracy_std": r.get("accuracy_std")}
                          for r in m.get("all_model_results") or []),
                         key=lambda r: -r["accuracy"]),
        "per_class": _per_class(pred) if has_pred else [],
        "confusion": _confusion(pred, labels) if has_pred else [],
        "confusion_labels": [EVENT_NAMES.get(e, e) for e in labels],
        "mistakes": _mistakes(pred) if has_pred else [],
        "top_features": top_features,
        "command": " \\\n    ".join(command),
    }
    return data


if __name__ == "__main__":
    import pprint
    pprint.pprint(load_training_dashboard(db_path=REPO_ROOT / "db" / "droneai.sqlite"))
