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
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC


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


def resolve_path(path_like):
    p = Path(path_like)
    if p.is_absolute():
        return p
    return PROJECT_DIR / p


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

        rows.append(out)

    features = pd.DataFrame(rows)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features


def make_split(features_df, split_mode, seed, val_size=0.25):
    y = features_df["label"].to_numpy()

    if split_mode == "clip":
        indices = np.arange(len(features_df))
        train_idx, val_idx = train_test_split(
            indices,
            test_size=val_size,
            random_state=seed,
            stratify=y,
        )
        return train_idx, val_idx

    if split_mode != "session":
        raise ValueError("split_mode must be clip or session")

    groups = features_df["session_name"].astype(str).to_numpy()
    labels = features_df["label"].astype(str).to_numpy()
    all_classes = set(labels)

    best = None
    best_score = 1e9

    # Try several group splits and prefer one where validation has all classes.
    for i in range(300):
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed + i)
        tr, va = next(splitter.split(features_df, labels, groups))

        train_classes = set(labels[tr])
        val_classes = set(labels[va])
        missing_penalty = 1000 * (len(all_classes - train_classes) + len(all_classes - val_classes))
        size_penalty = abs(len(va) / len(features_df) - val_size)
        score = missing_penalty + size_penalty

        if score < best_score:
            best = (tr, va)
            best_score = score

        if missing_penalty == 0 and size_penalty < 0.08:
            return tr, va

    return best


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-csv", default="LabelGUI/OpticalFlowResults/dpflow_drone_v1_gpu/flow_sequence_features.csv")
    parser.add_argument("--split-mode", choices=["clip", "session"], default="clip")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-size", type=float, default=0.25)
    args = parser.parse_args()

    features_csv = resolve_path(args.features_csv)
    if not features_csv.exists():
        raise FileNotFoundError(f"Could not find sequence features CSV: {features_csv}")

    run_name = args.run_name.strip() or f"dpflow_tabular_{args.split_mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = TRAINING_RUNS_DIR / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== DroneAI DPFlow Tabular Model Training ===")
    print("Features CSV:", features_csv)
    print("Split mode:", args.split_mode)
    print("Run name:", run_name)
    print("Output:", output_dir)

    seq_df = pd.read_csv(features_csv)
    clip_df = build_clip_features(seq_df)
    clip_df.to_csv(output_dir / "tabular_clip_features.csv", index=False)

    labels = sorted(clip_df["label"].astype(str).unique().tolist())
    label_encoder = LabelEncoder().fit(labels)

    metadata_cols = ["clip_group", "session_name", "clip_filename", "label"]
    feature_cols = [c for c in clip_df.columns if c not in metadata_cols]

    X = clip_df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = clip_df["label"].astype(str).to_numpy()

    train_idx, val_idx = make_split(clip_df, args.split_mode, args.seed, args.val_size)

    X_train = X.iloc[train_idx]
    y_train = y[train_idx]
    X_val = X.iloc[val_idx]
    y_val = y[val_idx]

    split_df = clip_df[["clip_group", "session_name", "clip_filename", "label"]].copy()
    split_df["split"] = "train"
    split_df.loc[val_idx, "split"] = "val"
    split_df.to_csv(output_dir / "split.csv", index=False)

    print("Total clips:", len(clip_df))
    print("Train clips:", len(train_idx))
    print("Val clips:", len(val_idx))
    print("Label counts:")
    print(clip_df["label"].value_counts().to_string())

    results = []
    best_name = None
    best_score = (-1.0, -1.0)
    best_eval = None
    best_importance = None

    models = get_models(args.seed)

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        ev = evaluate_model(model, model_name, X_train, y_train, X_val, y_val, labels, label_encoder=label_encoder)

        print(f"{model_name}: accuracy={ev['accuracy']:.4f}, macro_f1={ev['macro_f1']:.4f}, weighted_f1={ev['weighted_f1']:.4f}")

        results.append({
            "model": model_name,
            "accuracy": ev["accuracy"],
            "macro_f1": ev["macro_f1"],
            "weighted_f1": ev["weighted_f1"],
        })

        score = (ev["accuracy"], ev["macro_f1"])
        if score > best_score:
            best_score = score
            best_name = model_name
            best_eval = ev
            best_importance = feature_importance(ev["model"], model_name, feature_cols, label_encoder)

    results_df = pd.DataFrame(results).sort_values(["accuracy", "macro_f1"], ascending=False)
    results_df.to_csv(output_dir / "all_model_results.csv", index=False)

    # Save best predictions.
    pred_df = clip_df.iloc[val_idx][["clip_group", "session_name", "clip_filename", "label"]].copy()
    pred_df = pred_df.rename(columns={"label": "true_label"})
    pred_df["pred_label"] = best_eval["y_pred"]
    pred_df["correct"] = pred_df["true_label"] == pred_df["pred_label"]

    if best_eval["proba"] is not None:
        proba = np.asarray(best_eval["proba"])
        # XGBoost proba classes follow label_encoder order; sklearn follows model.classes_.
        if best_name == "xgboost":
            proba_labels = label_encoder.classes_.tolist()
        else:
            fitted = best_eval["model"]
            if isinstance(fitted, Pipeline):
                fitted = fitted.named_steps.get("model", fitted)
            proba_labels = list(getattr(fitted, "classes_", labels))

        for i, lab in enumerate(proba_labels):
            if i < proba.shape[1]:
                pred_df[f"prob_{lab}"] = proba[:, i]

    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    cm_df = pd.DataFrame(best_eval["confusion_matrix"], index=labels, columns=labels)
    cm_df.to_csv(output_dir / "confusion_matrix.csv")

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(best_eval["report_text"])

    if best_importance is not None and len(best_importance):
        best_importance.to_csv(output_dir / "feature_importance.csv", index=False)
    else:
        pd.DataFrame(columns=["feature", "importance", "source"]).to_csv(output_dir / "feature_importance.csv", index=False)

    metrics = {
        "run_name": run_name,
        "stage": "tabular_training",
        "input_features_csv": str(features_csv),
        "split_mode": args.split_mode,
        "seed": args.seed,
        "total_clips": int(len(clip_df)),
        "train_clips": int(len(train_idx)),
        "val_clips": int(len(val_idx)),
        "labels": labels,
        "label_counts": clip_df["label"].value_counts().to_dict(),
        "feature_count": int(len(feature_cols)),
        "best_model": best_name,
        "accuracy": best_eval["accuracy"],
        "macro_f1": best_eval["macro_f1"],
        "weighted_f1": best_eval["weighted_f1"],
        "all_model_results": results,
        "notes": "DPFlow sequence features summarized into one tabular feature vector per clip. Tree/linear/SVM baselines trained for small-data comparison against LSTM and VideoMAE.",
        "output_dir": str(output_dir),
    }

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=json_safe)

    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "script": "train_dpflow_tabular_models.py",
            "purpose": "Feature-engineered tabular baseline from corrected DPFlow motion sequence features.",
            "models_trained": list(models.keys()),
            "best_model": best_name,
            "feature_columns": feature_cols,
        }, f, indent=2, default=json_safe)

    with open(output_dir / "notes.txt", "w", encoding="utf-8") as f:
        f.write(
            "DPFlow tabular model experiment.\n"
            "This run summarizes each clip into physical motion statistics such as speed peaks, acceleration, vertical motion, ROI size, and confidence.\n"
            "Keep this folder for paper result tracking.\n"
        )

    with open(output_dir / "best_model.pkl", "wb") as f:
        pickle.dump(best_eval["model"], f)

    append_registry({
        "date": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "stage": "tabular_training",
        "dataset": "DroneAI current labeled clips",
        "flow_method": "DPFlow",
        "model": best_name,
        "split_type": args.split_mode,
        "train_clips": len(train_idx),
        "val_clips": len(val_idx),
        "test_clips": 0,
        "accuracy": best_eval["accuracy"],
        "macro_f1": best_eval["macro_f1"],
        "weighted_f1": best_eval["weighted_f1"],
        "best_epoch": "n/a",
        "notes": "DPFlow tabular summary feature baseline.",
        "result_folder": str(output_dir),
    })

    print("\n=== Done ===")
    print("Best model:", best_name)
    print(f"Accuracy: {best_eval['accuracy']:.4f}")
    print(f"Macro F1: {best_eval['macro_f1']:.4f}")
    print(f"Weighted F1: {best_eval['weighted_f1']:.4f}")
    print("\nAll model results:")
    print(results_df.to_string(index=False))
    print("\nClassification report for best model:")
    print(best_eval["report_text"])
    print("\nSaved to:", output_dir)


if __name__ == "__main__":
    main()
