import argparse
import json
import math
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC

from repo_paths import repo_rel, resolve_path


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
TRAINING_RUNS_DIR = BASE_DIR / "TrainingRuns"
EPS = 1e-9


def json_safe(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def safe_series(g, col):
    if col not in g.columns:
        return np.zeros(len(g), dtype=float)
    return pd.to_numeric(g[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)


def add_basic_stats(out, prefix, values, dt_values=None):
    values = np.asarray(values, dtype=float)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    if len(values) == 0:
        values = np.array([0.0], dtype=float)

    out[f"{prefix}_mean"] = float(np.mean(values))
    out[f"{prefix}_std"] = float(np.std(values))
    out[f"{prefix}_min"] = float(np.min(values))
    out[f"{prefix}_max"] = float(np.max(values))
    out[f"{prefix}_median"] = float(np.median(values))
    out[f"{prefix}_p10"] = float(np.percentile(values, 10))
    out[f"{prefix}_p25"] = float(np.percentile(values, 25))
    out[f"{prefix}_p75"] = float(np.percentile(values, 75))
    out[f"{prefix}_p90"] = float(np.percentile(values, 90))
    out[f"{prefix}_range"] = float(np.max(values) - np.min(values))
    out[f"{prefix}_first"] = float(values[0])
    out[f"{prefix}_last"] = float(values[-1])
    out[f"{prefix}_delta"] = float(values[-1] - values[0])
    out[f"{prefix}_abs_mean"] = float(np.mean(np.abs(values)))
    out[f"{prefix}_abs_max"] = float(np.max(np.abs(values)))
    out[f"{prefix}_energy"] = float(np.sum(values ** 2))
    out[f"{prefix}_peak_to_mean"] = float(np.max(np.abs(values)) / (np.mean(np.abs(values)) + EPS))

    if len(values) > 1:
        peak_idx = int(np.argmax(np.abs(values)))
        out[f"{prefix}_time_to_abs_peak"] = float(peak_idx / max(len(values) - 1, 1))
        out[f"{prefix}_slope_first_last"] = float((values[-1] - values[0]) / max(len(values) - 1, 1))
    else:
        out[f"{prefix}_time_to_abs_peak"] = 0.0
        out[f"{prefix}_slope_first_last"] = 0.0

    if dt_values is not None and len(dt_values) == len(values):
        dt_values = np.nan_to_num(np.asarray(dt_values, dtype=float), nan=0.1, posinf=0.1, neginf=0.1)
        dt_values = np.clip(dt_values, 1e-6, None)
        out[f"{prefix}_auc"] = float(np.sum(np.abs(values) * dt_values))
        out[f"{prefix}_signed_auc"] = float(np.sum(values * dt_values))
    else:
        out[f"{prefix}_auc"] = float(np.sum(np.abs(values)))
        out[f"{prefix}_signed_auc"] = float(np.sum(values))


def add_derivative_stats(out, prefix, values, dt_values):
    values = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    dt_values = np.nan_to_num(np.asarray(dt_values, dtype=float), nan=0.1, posinf=0.1, neginf=0.1)
    dt_values = np.clip(dt_values, 1e-6, None)

    if len(values) < 2:
        deriv = np.array([0.0], dtype=float)
    else:
        # Use dt at the later step for each difference.
        deriv = np.diff(values) / dt_values[1:]

    add_basic_stats(out, prefix, deriv)

    if len(deriv) > 1:
        jerk = np.diff(deriv)
    else:
        jerk = np.array([0.0], dtype=float)

    add_basic_stats(out, f"{prefix}_jerk", jerk)


def build_clip_features(sequence_df):
    required = ["clip_group", "label", "step_index"]
    for col in required:
        if col not in sequence_df.columns:
            raise ValueError(f"Missing required column in sequence CSV: {col}")

    rows = []

    for clip_group, g in sequence_df.groupby("clip_group"):
        g = g.sort_values("step_index").copy()
        label = str(g["label"].iloc[0])
        session_name = str(g["session_name"].iloc[0]) if "session_name" in g.columns else "unknown_session"
        clip_filename = str(g["clip_filename"].iloc[0]) if "clip_filename" in g.columns else str(clip_group)

        out = {
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_filename,
            "label": label,
            "video_id": str(g["video_id"].iloc[0]) if "video_id" in g.columns else "",
            "view_type": str(g["view_type"].iloc[0]) if "view_type" in g.columns else "",
            "num_steps": int(len(g)),
        }

        dt = safe_series(g, "dt")
        if len(dt) == 0:
            dt = np.array([0.1], dtype=float)
        dt = np.clip(dt, 1e-6, None)

        out["duration_estimate_sec"] = float(np.sum(dt))
        out["mean_dt"] = float(np.mean(dt))

        # Original step-level signals from DPFlow extraction.
        signal_cols = [
            "roi_available",
            "both_detected",
            "any_detected",
            "flow_dx_norm_per_sec",
            "flow_dy_norm_per_sec",
            "flow_mag_norm_per_sec",
            "flow_dx_mean_per_sec",
            "flow_dy_mean_per_sec",
            "flow_mag_mean_per_sec",
            "flow_mag_mean",
            "flow_mag_median",
            "flow_mag_max",
            "flow_mag_std",
            "det_vx_norm_per_sec",
            "det_vy_norm_per_sec",
            "det_speed_norm_per_sec",
            "det_dx",
            "det_dy",
            "det_speed",
            "conf_a",
            "conf_b",
            "roi_width",
            "roi_height",
            # Whole-frame ego-motion (FPV); zeros for feature files made before it existed.
            "frame_dx_norm_per_sec",
            "frame_dy_norm_per_sec",
            "frame_mag_norm_per_sec",
            "frame_mag_std_norm_per_sec",
            "frame_mag_p90_norm_per_sec",
            "frame_divergence_per_sec",
            "frame_curl_per_sec",
            "frame_still_fraction",
            "frame_coherence",
        ]

        for col in signal_cols:
            vals = safe_series(g, col)
            add_basic_stats(out, col, vals, dt_values=dt)

        # Derived geometric/physics-style signals.
        roi_w = safe_series(g, "roi_width")
        roi_h = safe_series(g, "roi_height")
        roi_area = roi_w * roi_h
        roi_aspect = roi_w / (roi_h + EPS)
        add_basic_stats(out, "roi_area", roi_area, dt_values=dt)
        add_basic_stats(out, "roi_aspect", roi_aspect, dt_values=dt)

        det_speed = safe_series(g, "det_speed_norm_per_sec")
        flow_mag = safe_series(g, "flow_mag_norm_per_sec")
        det_vy = safe_series(g, "det_vy_norm_per_sec")
        det_vx = safe_series(g, "det_vx_norm_per_sec")
        flow_dy = safe_series(g, "flow_dy_norm_per_sec")
        flow_dx = safe_series(g, "flow_dx_norm_per_sec")

        add_derivative_stats(out, "det_speed_accel", det_speed, dt)
        add_derivative_stats(out, "flow_mag_accel", flow_mag, dt)
        add_derivative_stats(out, "det_vy_accel", det_vy, dt)
        add_derivative_stats(out, "flow_dy_accel", flow_dy, dt)

        # Direction-specific cues. In image coordinates, positive y is generally downward.
        out["det_max_downward_vy"] = float(np.max(det_vy)) if len(det_vy) else 0.0
        out["det_max_upward_vy"] = float(abs(np.min(det_vy))) if len(det_vy) else 0.0
        out["flow_max_downward_dy"] = float(np.max(flow_dy)) if len(flow_dy) else 0.0
        out["flow_max_upward_dy"] = float(abs(np.min(flow_dy))) if len(flow_dy) else 0.0
        out["det_down_up_ratio"] = float(out["det_max_downward_vy"] / (out["det_max_upward_vy"] + EPS))
        out["flow_down_up_ratio"] = float(out["flow_max_downward_dy"] / (out["flow_max_upward_dy"] + EPS))

        # Horizontal vs vertical dominance.
        out["det_vertical_horizontal_ratio"] = float(np.mean(np.abs(det_vy)) / (np.mean(np.abs(det_vx)) + EPS))
        out["flow_vertical_horizontal_ratio"] = float(np.mean(np.abs(flow_dy)) / (np.mean(np.abs(flow_dx)) + EPS))

        # Start/end behavior. Helpful for takeoff vs landing.
        if len(det_speed) >= 3:
            k = max(1, len(det_speed) // 3)
            out["det_speed_start_mean"] = float(np.mean(det_speed[:k]))
            out["det_speed_mid_mean"] = float(np.mean(det_speed[k:2*k])) if len(det_speed[k:2*k]) else 0.0
            out["det_speed_end_mean"] = float(np.mean(det_speed[-k:]))
            out["flow_mag_start_mean"] = float(np.mean(flow_mag[:k]))
            out["flow_mag_mid_mean"] = float(np.mean(flow_mag[k:2*k])) if len(flow_mag[k:2*k]) else 0.0
            out["flow_mag_end_mean"] = float(np.mean(flow_mag[-k:]))
        else:
            out["det_speed_start_mean"] = float(np.mean(det_speed)) if len(det_speed) else 0.0
            out["det_speed_mid_mean"] = 0.0
            out["det_speed_end_mean"] = float(np.mean(det_speed)) if len(det_speed) else 0.0
            out["flow_mag_start_mean"] = float(np.mean(flow_mag)) if len(flow_mag) else 0.0
            out["flow_mag_mid_mean"] = 0.0
            out["flow_mag_end_mean"] = float(np.mean(flow_mag)) if len(flow_mag) else 0.0

        out["det_speed_end_minus_start"] = out["det_speed_end_mean"] - out["det_speed_start_mean"]
        out["flow_mag_end_minus_start"] = out["flow_mag_end_mean"] - out["flow_mag_start_mean"]

        # Whole-frame: how motion changes over the clip (takeoff speeds up, landing
        # and crashes end still; a crash is a sudden spike).
        for name in ("frame_mag_norm_per_sec", "frame_still_fraction",
                     "frame_divergence_per_sec", "frame_coherence"):
            vals = safe_series(g, name)
            k = max(1, len(vals) // 3)
            start, end = float(np.mean(vals[:k])), float(np.mean(vals[-k:]))
            out[f"{name}_start_mean"] = start
            out[f"{name}_end_mean"] = end
            out[f"{name}_end_minus_start"] = end - start
        add_derivative_stats(out, "frame_mag_accel", safe_series(g, "frame_mag_norm_per_sec"), dt)

        rows.append(out)

    features = pd.DataFrame(rows)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


METADATA_COLS = ["clip_group", "session_name", "clip_filename", "label", "video_id", "view_type"]


def attach_video_ids(clip_df, video_map_path):
    """
    Fill clip_df["video_id"] and clip_df["view_type"].

    Order: a video_id column already in the features CSV, then the session-to-video
    map written by `python LabelGUI/video_manifest.py` (session_name -> video_id).
    Clips still unmapped are grouped by their session folder name, which is weaker:
    a video labeled twice gets two folders, so its clips could land on both sides.

    Returns (clip_df, info dict for metrics.json).
    """
    clip_df = clip_df.copy()
    clip_df["video_id"] = clip_df["video_id"].fillna("").astype(str).replace("nan", "")
    clip_df["view_type"] = clip_df["view_type"].fillna("").astype(str).replace("nan", "")
    from_column = int((clip_df["video_id"] != "").sum())

    from_map = 0
    map_used = ""
    if video_map_path is not None and Path(video_map_path).exists():
        vmap = pd.read_csv(video_map_path, dtype=str).fillna("")
        map_used = repo_rel(video_map_path)
        vmap = vmap.drop_duplicates(subset=["session_name"], keep="last").set_index("session_name")
        need = clip_df["video_id"] == ""
        mapped = clip_df.loc[need, "session_name"].map(vmap["video_id"]).fillna("")
        clip_df.loc[need, "video_id"] = mapped
        from_map = int((mapped != "").sum())
        if "view_type" in vmap.columns:
            need_view = clip_df["view_type"] == ""
            clip_df.loc[need_view, "view_type"] = (
                clip_df.loc[need_view, "session_name"].map(vmap["view_type"]).fillna("")
            )

    unmapped = clip_df["video_id"] == ""
    fallback = int(unmapped.sum())
    clip_df.loc[unmapped, "video_id"] = "session:" + clip_df.loc[unmapped, "session_name"].astype(str)
    clip_df.loc[clip_df["view_type"] == "", "view_type"] = "Unknown"

    info = {
        "video_map": map_used,
        "clips_video_id_from_features": from_column,
        "clips_video_id_from_map": from_map,
        "clips_grouped_by_session_fallback": fallback,
    }
    return clip_df, info


def load_excluded_videos(path):
    """video_ids never to train or validate on (the held-out test split)."""
    if path is None or not Path(path).exists():
        return set()
    df = pd.read_csv(path, dtype=str)
    if "video_id" not in df.columns:
        raise ValueError(f"{path} needs a video_id column")
    return set(df["video_id"].dropna().astype(str))


def make_group_folds(clip_df, n_folds, seed):
    """
    Grouped, label-stratified K folds: all clips of one video are in the same fold.
    Returns a list of (train_idx, val_idx) and the per-clip fold number.
    """
    groups = clip_df["video_id"].astype(str).to_numpy()
    labels = clip_df["label"].astype(str).to_numpy()
    n_groups = len(set(groups))
    if n_groups < 2:
        raise ValueError(f"Need clips from at least 2 videos for grouped CV, found {n_groups}.")
    # StratifiedGroupKFold refuses when every class has fewer clips than folds.
    biggest_class = int(pd.Series(labels).value_counts().max())
    k = min(n_folds, n_groups, biggest_class)
    if k < 2:
        raise ValueError("Need at least 2 clips of some class for grouped CV.")
    if k < n_folds:
        print(f"WARNING: only {n_groups} videos and at most {biggest_class} clips per class, "
              f"using {k} folds instead of {n_folds}.")

    splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    fold_of = np.full(len(clip_df), -1, dtype=int)
    for i, (tr, va) in enumerate(splitter.split(clip_df, labels, groups)):
        folds.append((tr, va))
        fold_of[va] = i
    return folds, fold_of


def get_models(seed):
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=700,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=900,
            random_state=seed,
            class_weight="balanced",
            min_samples_leaf=1,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=seed),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed)),
        ]),
        "svm_rbf": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
        ]),
    }

    # Optional XGBoost if installed. It is skipped automatically if not available.
    try:
        from xgboost import XGBClassifier
        models["xgboost"] = XGBClassifier(
            n_estimators=350,
            max_depth=3,
            learning_rate=0.035,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=seed,
            n_jobs=-1,
        )
    except Exception:
        pass

    return models


def feature_importance(model, model_name, feature_names, label_encoder=None):
    fitted = model
    if isinstance(model, Pipeline):
        fitted = model.named_steps.get("model", model)

    rows = []

    if hasattr(fitted, "feature_importances_"):
        vals = np.asarray(fitted.feature_importances_, dtype=float)
        for name, val in zip(feature_names, vals):
            rows.append({"feature": name, "importance": float(val), "source": model_name})
    elif hasattr(fitted, "coef_"):
        vals = np.mean(np.abs(np.asarray(fitted.coef_, dtype=float)), axis=0)
        for name, val in zip(feature_names, vals):
            rows.append({"feature": name, "importance": float(val), "source": model_name})

    rows = sorted(rows, key=lambda r: r["importance"], reverse=True)
    return pd.DataFrame(rows)


def evaluate_model(model, model_name, X_train, y_train, X_val, y_val, labels, label_encoder=None):
    if model_name == "xgboost":
        y_train_enc = label_encoder.transform(y_train)
        model.fit(X_train, y_train_enc)
        pred_enc = model.predict(X_val)
        y_pred = label_encoder.inverse_transform(pred_enc.astype(int))
        proba = model.predict_proba(X_val)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_val, y_pred)
    macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    report_dict = classification_report(y_val, y_pred, labels=labels, zero_division=0, output_dict=True)
    report_text = classification_report(y_val, y_pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y_val, y_pred, labels=labels)

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "y_pred": y_pred,
        "proba": proba,
        "report_dict": report_dict,
        "report_text": report_text,
        "confusion_matrix": cm,
        "model": model,
    }


def append_registry(row):
    registry_path = TRAINING_RUNS_DIR / "experiment_registry.csv"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([row])

    if registry_path.exists():
        old = pd.read_csv(registry_path)
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df

    out.to_csv(registry_path, index=False)


def summarize(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def run_model_cv(model_name, model, X, y, folds, labels, label_encoder):
    """Train a fresh copy per fold. Returns per-fold scores and out-of-fold predictions."""
    per_fold = []
    oof_pred = np.empty(len(y), dtype=object)
    oof_proba = None
    for i, (tr, va) in enumerate(folds):
        ev = evaluate_model(clone(model), model_name, X.iloc[tr], y[tr], X.iloc[va], y[va],
                            labels, label_encoder=label_encoder)
        per_fold.append({
            "fold": i,
            "val_clips": int(len(va)),
            "val_labels": sorted(set(y[va].tolist())),
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
        })
        oof_pred[va] = ev["y_pred"]
        if ev["proba"] is not None:
            classes = proba_classes(ev["model"], model_name, labels, label_encoder)
            if oof_proba is None:
                oof_proba = np.zeros((len(y), len(labels)), dtype=float)
            proba = np.asarray(ev["proba"])
            for j, lab in enumerate(classes):
                if lab in labels and j < proba.shape[1]:
                    oof_proba[va, labels.index(lab)] = proba[:, j]
    return per_fold, oof_pred, oof_proba


def proba_classes(fitted, model_name, labels, label_encoder):
    # XGBoost proba classes follow label_encoder order; sklearn follows model.classes_.
    if model_name == "xgboost":
        return label_encoder.classes_.tolist()
    if isinstance(fitted, Pipeline):
        fitted = fitted.named_steps.get("model", fitted)
    return [str(c) for c in getattr(fitted, "classes_", labels)]


def main():
    parser = argparse.ArgumentParser(
        description="DPFlow tabular baselines with grouped K-fold cross-validation by video_id."
    )
    parser.add_argument("--features-csv", default="LabelGUI/OpticalFlowResults/dpflow_drone_v1_gpu/flow_sequence_features.csv")
    parser.add_argument("--video-map", default="analysis/data/session_videos.csv",
                        help="session_name -> video_id map from `python LabelGUI/video_manifest.py`. "
                             "Not needed when the features CSV already has a video_id column.")
    parser.add_argument("--view-type", choices=["all", "FPV", "Third-person"], default="all",
                        help="Only use clips from videos with this view type.")
    parser.add_argument("--exclude-videos", default="analysis/data/test_videos.csv",
                        help="CSV with a video_id column: the held-out test videos, never used here.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    features_csv = resolve_path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Could not find sequence features CSV: {features_csv}")
    # Features made from a data snapshot: use that snapshot's session map and
    # test split too (unless given explicitly), so the whole run is one dataset.
    data_snapshot = None
    try:
        data_snapshot = json.loads((features_csv.parent / "run_summary.json").read_text(encoding="utf-8")).get("data_snapshot")
    except (OSError, ValueError):
        pass
    if data_snapshot and data_snapshot.get("dir"):
        snap_dir = resolve_path(data_snapshot["dir"])
        if args.video_map == parser.get_default("video_map") and (snap_dir / "session_videos.csv").exists():
            args.video_map = str(snap_dir / "session_videos.csv")
        if args.exclude_videos == parser.get_default("exclude_videos") and (snap_dir / "test_videos.csv").exists():
            args.exclude_videos = str(snap_dir / "test_videos.csv")
    video_map = resolve_path(args.video_map)
    exclude_path = resolve_path(args.exclude_videos)

    run_name = args.run_name.strip() or f"dpflow_tabular_groupcv_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = TRAINING_RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== DroneAI DPFlow Tabular Model Training (grouped CV by video) ===")
    print("Features CSV:", features_csv)
    print("Data snapshot:", f"{data_snapshot['id']} ({data_snapshot['fingerprint'][:12]})" if data_snapshot else "none (live data)")
    print("Video map:", video_map if video_map and video_map.exists() else "(none)")
    print("View type filter:", args.view_type)
    print("Run name:", run_name)
    print("Output:", output_dir)

    seq_df = pd.read_csv(features_csv)
    clip_df = build_clip_features(seq_df)
    clip_df, group_info = attach_video_ids(clip_df, video_map)

    if group_info["clips_grouped_by_session_fallback"]:
        print(f"WARNING: {group_info['clips_grouped_by_session_fallback']} clips have no video_id; "
              "they are grouped by session folder instead. Run `python LabelGUI/video_manifest.py` "
              "to write analysis/data/session_videos.csv.")

    total_before = len(clip_df)
    excluded = load_excluded_videos(exclude_path)
    n_excluded_clips = int(clip_df["video_id"].isin(excluded).sum())
    clip_df = clip_df[~clip_df["video_id"].isin(excluded)]
    if excluded:
        print(f"Held-out test videos: {len(excluded)} ({n_excluded_clips} clips removed)")

    if args.view_type != "all":
        clip_df = clip_df[clip_df["view_type"] == args.view_type]
        print(f"Clips with view type {args.view_type}: {len(clip_df)}")
    clip_df = clip_df.reset_index(drop=True)
    if clip_df.empty:
        raise ValueError("No clips left after filtering.")
    clip_df.to_csv(output_dir / "tabular_clip_features.csv", index=False)

    labels = sorted(clip_df["label"].astype(str).unique().tolist())
    label_encoder = LabelEncoder().fit(labels)

    feature_cols = [c for c in clip_df.columns if c not in METADATA_COLS]
    X = clip_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = clip_df["label"].astype(str).to_numpy()

    folds, fold_of = make_group_folds(clip_df, args.folds, args.seed)
    n_videos = int(clip_df["video_id"].nunique())

    folds_df = clip_df[["clip_group", "session_name", "clip_filename", "video_id", "view_type", "label"]].copy()
    folds_df["fold"] = fold_of
    folds_df.to_csv(output_dir / "folds.csv", index=False)

    print("Total clips:", len(clip_df), "| videos:", n_videos, "| folds:", len(folds))
    print("Label counts:")
    print(clip_df["label"].value_counts().to_string())

    results = []
    per_fold_all = {}
    best = None
    models = get_models(args.seed)

    for model_name, model in models.items():
        print(f"\nTraining {model_name} ({len(folds)} folds)...")
        try:
            per_fold, oof_pred, oof_proba = run_model_cv(model_name, model, X, y, folds, labels, label_encoder)
        except Exception as e:  # e.g. xgboost when a fold's training part misses a class
            print(f"WARNING: skipping {model_name}: {e}")
            continue
        acc_m, acc_s = summarize([f["accuracy"] for f in per_fold])
        mac_m, mac_s = summarize([f["macro_f1"] for f in per_fold])
        wei_m, wei_s = summarize([f["weighted_f1"] for f in per_fold])
        print(f"{model_name}: accuracy={acc_m:.4f} +- {acc_s:.4f}, macro_f1={mac_m:.4f} +- {mac_s:.4f}")

        per_fold_all[model_name] = per_fold
        results.append({
            "model": model_name,
            "accuracy": acc_m, "accuracy_std": acc_s,
            "macro_f1": mac_m, "macro_f1_std": mac_s,
            "weighted_f1": wei_m, "weighted_f1_std": wei_s,
        })
        score = (acc_m, mac_m)
        if best is None or score > best["score"]:
            best = {"name": model_name, "score": score, "oof_pred": oof_pred, "oof_proba": oof_proba,
                    "result": results[-1]}

    best_name = best["name"]
    results_df = pd.DataFrame(results).sort_values(["accuracy", "macro_f1"], ascending=False)
    results_df.to_csv(output_dir / "all_model_results.csv", index=False)

    fold_rows = [dict(model=m, **f) for m, fs in per_fold_all.items() for f in fs]
    pd.DataFrame(fold_rows).drop(columns=["val_labels"]).to_csv(output_dir / "fold_results.csv", index=False)

    # Out-of-fold predictions of the best model: every clip predicted by a model
    # that never saw its video.
    pred_df = folds_df.rename(columns={"label": "true_label"})
    pred_df["pred_label"] = best["oof_pred"]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]
    if best["oof_proba"] is not None:
        for j, lab in enumerate(labels):
            pred_df[f"prob_{lab}"] = best["oof_proba"][:, j]
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    oof_pred = best["oof_pred"].astype(str)
    cm = confusion_matrix(y, oof_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(output_dir / "confusion_matrix.csv")
    report_text = classification_report(y, oof_pred, labels=labels, zero_division=0)
    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"Out-of-fold predictions, {best_name}, {len(folds)}-fold grouped by video_id\n\n")
        f.write(report_text)

    # Final model: the best model trained on all clips (for later use), with its importances.
    final_model = clone(models[best_name])
    if best_name == "xgboost":
        final_model.fit(X, label_encoder.transform(y))
    else:
        final_model.fit(X, y)
    importance = feature_importance(final_model, best_name, feature_cols, label_encoder)
    if importance is not None and len(importance):
        importance.to_csv(output_dir / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance", "source"]).to_csv(output_dir / "feature_importance.csv", index=False)
    with open(output_dir / "best_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    br = best["result"]
    metrics = {
        "run_name": run_name,
        "stage": "tabular_training",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_snapshot": data_snapshot,
        "input_features_csv": repo_rel(features_csv),
        "split_mode": "grouped_kfold_video",
        "folds": len(folds),
        "seed": args.seed,
        "view_type_filter": args.view_type,
        "held_out_test_videos_file": repo_rel(exclude_path) if excluded else "",
        "held_out_test_videos": len(excluded),
        "held_out_clips_removed": n_excluded_clips,
        "clips_before_filters": int(total_before),
        "total_clips": int(len(clip_df)),
        "total_videos": n_videos,
        "grouping": group_info,
        "labels": labels,
        "label_counts": clip_df["label"].value_counts().to_dict(),
        "feature_count": int(len(feature_cols)),
        "best_model": best_name,
        "accuracy": br["accuracy"],
        "accuracy_std": br["accuracy_std"],
        "macro_f1": br["macro_f1"],
        "macro_f1_std": br["macro_f1_std"],
        "weighted_f1": br["weighted_f1"],
        "weighted_f1_std": br["weighted_f1_std"],
        "per_fold_best_model": per_fold_all[best_name],
        "all_model_results": results,
        "notes": "DPFlow sequence features summarized into one tabular vector per clip. Scores are mean +- std "
                 "over K folds grouped by video_id (no video in both train and validation). best_model.pkl is the "
                 "best model refit on all clips.",
        "output_dir": repo_rel(output_dir),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=json_safe)

    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "script": "train_dpflow_tabular_models.py",
            "purpose": "Feature-engineered tabular baseline from DPFlow motion sequence features, grouped CV by video.",
            "models_trained": list(models.keys()),
            "best_model": best_name,
            "feature_columns": feature_cols,
        }, f, indent=2, default=json_safe)

    with open(output_dir / "notes.txt", "w", encoding="utf-8") as f:
        f.write(
            "DPFlow tabular model experiment, grouped K-fold cross-validation by video_id.\n"
            "Each clip is summarized into motion statistics such as speed peaks, acceleration, vertical motion, ROI size, and confidence.\n"
            "Keep this folder for paper result tracking.\n"
        )

    append_registry({
        "date": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "stage": "tabular_training",
        "dataset": "DroneAI current labeled clips",
        "flow_method": "DPFlow",
        "model": best_name,
        "split_type": f"grouped_{len(folds)}fold_video",
        "train_clips": int(len(clip_df)),
        "val_clips": int(len(clip_df)),
        "test_clips": 0,
        "accuracy": br["accuracy"],
        "accuracy_std": br["accuracy_std"],
        "macro_f1": br["macro_f1"],
        "macro_f1_std": br["macro_f1_std"],
        "weighted_f1": br["weighted_f1"],
        "best_epoch": "n/a",
        "notes": f"DPFlow tabular baseline, view={args.view_type}, {n_videos} videos.",
        "result_folder": repo_rel(output_dir),
    })

    print("\n=== Done ===")
    print(f"Best model: {best_name}")
    print(f"Accuracy: {br['accuracy']:.4f} +- {br['accuracy_std']:.4f}")
    print(f"Macro F1: {br['macro_f1']:.4f} +- {br['macro_f1_std']:.4f}")
    print(f"Weighted F1: {br['weighted_f1']:.4f} +- {br['weighted_f1_std']:.4f}")
    print("\nAll model results (mean over folds):")
    print(results_df.to_string(index=False))
    print("\nOut-of-fold classification report for the best model:")
    print(report_text)
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
