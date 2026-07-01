import argparse
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FEATURES_CSV = BASE_DIR / "MotionResults" / "motion_v2_all" / "clip_motion_features.csv"
RESULTS_BASE_DIR = BASE_DIR / "MotionResults"


ID_COLUMNS = {
    "clip_group",
    "session_name",
    "clip_filename",
    "label",
}


def load_features(features_csv: Path):
    if not features_csv.exists():
        raise FileNotFoundError(f"Could not find: {features_csv}")

    df = pd.read_csv(features_csv)

    if "label" not in df.columns:
        raise ValueError("clip_motion_features.csv must have a label column.")

    feature_cols = []

    for col in df.columns:
        if col in ID_COLUMNS:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No numeric motion feature columns found.")

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    y = df["label"].astype(str)

    return df, X, y, feature_cols


def build_model(model_name: str):
    if model_name == "rf":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=700,
            random_state=42,
            class_weight="balanced",
            min_samples_leaf=2,
        )

    if model_name == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=42,
            ),
        )

    raise ValueError(f"Unknown model: {model_name}")


def make_split(df, y, split_mode="clip", test_size=0.25):
    indices = np.arange(len(df))
    labels = sorted(y.unique().tolist())

    if split_mode == "clip":
        label_counts = y.value_counts()
        stratify = y if label_counts.min() >= 2 else None

        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        note = (
            "Clip-level random split. Each row is one clip. "
            "This is useful for a first motion-feature baseline."
        )

        return train_idx, val_idx, note

    if split_mode == "session":
        if "session_name" not in df.columns:
            raise ValueError("session split requires session_name column.")

        groups = df["session_name"].astype(str)

        best_train_idx = None
        best_val_idx = None

        # Try multiple seeds to find a split where all labels appear in both train and validation.
        for seed in range(42, 142):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=seed,
            )

            train_idx, val_idx = next(splitter.split(df, y, groups=groups))

            train_labels = set(y.iloc[train_idx].unique())
            val_labels = set(y.iloc[val_idx].unique())

            if set(labels).issubset(train_labels) and set(labels).issubset(val_labels):
                best_train_idx = train_idx
                best_val_idx = val_idx
                break

        if best_train_idx is None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=42,
            )
            best_train_idx, best_val_idx = next(splitter.split(df, y, groups=groups))

        note = (
            "Session-level split. Full source sessions/videos are kept together. "
            "This is stricter, but with the current small dataset it may be unstable."
        )

        return best_train_idx, best_val_idx, note

    raise ValueError(f"Unknown split mode: {split_mode}")


def safe_probabilities(model, X_val, labels):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_val)
        model_labels = list(model.classes_)

        rows = []

        for row_probs in probs:
            d = {}

            for label in labels:
                if label in model_labels:
                    idx = model_labels.index(label)
                    d[f"prob_{label}"] = float(row_probs[idx])
                else:
                    d[f"prob_{label}"] = 0.0

            rows.append(d)

        return rows

    return [{} for _ in range(len(X_val))]


def run_cross_validation(model, X, y, df):
    results = {}

    label_counts = y.value_counts()

    if label_counts.min() >= 5:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

        results["stratified_5fold_accuracy"] = {
            "scores": [float(s) for s in scores],
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }

    if "session_name" in df.columns and df["session_name"].nunique() >= 5:
        groups = df["session_name"].astype(str)
        n_splits = min(5, df["session_name"].nunique())

        try:
            gkf = GroupKFold(n_splits=n_splits)
            scores = cross_val_score(model, X, y, cv=gkf, groups=groups, scoring="accuracy")

            results["group_5fold_accuracy"] = {
                "scores": [float(s) for s in scores],
                "mean": float(scores.mean()),
                "std": float(scores.std()),
            }
        except Exception as e:
            results["group_5fold_error"] = str(e)

    return results


def get_feature_importance(model, feature_cols):
    actual_model = model

    if hasattr(model, "named_steps"):
        actual_model = list(model.named_steps.values())[-1]

    if hasattr(actual_model, "feature_importances_"):
        values = actual_model.feature_importances_
        rows = []

        for col, value in zip(feature_cols, values):
            rows.append({
                "feature": col,
                "importance": float(value),
            })

        rows = sorted(rows, key=lambda r: r["importance"], reverse=True)
        return rows

    return []


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--features-csv",
        default=str(DEFAULT_FEATURES_CSV),
        help="Path to clip_motion_features.csv",
    )

    parser.add_argument(
        "--model",
        default="rf",
        choices=["rf", "extra_trees", "logreg"],
        help="Classifier type",
    )

    parser.add_argument(
        "--split-mode",
        default="clip",
        choices=["clip", "session"],
        help="clip = random clip split, session = hold out full sessions",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
    )

    args = parser.parse_args()

    features_csv = Path(args.features_csv)

    df, X, y, feature_cols = load_features(features_csv)

    print("\n==============================")
    print("DroneAI Motion Classifier")
    print("==============================\n")

    print(f"[INFO] Features CSV: {features_csv}")
    print(f"[INFO] Total clips: {len(df)}")
    print(f"[INFO] Total features: {len(feature_cols)}")
    print(f"[INFO] Model: {args.model}")
    print(f"[INFO] Split mode: {args.split_mode}")

    print("\n[INFO] Clip counts by label:")
    for label, count in y.value_counts().items():
        print(f"  - {label}: {count}")

    if "session_name" in df.columns:
        print(f"\n[INFO] Source sessions: {df['session_name'].nunique()}")

    train_idx, val_idx, split_note = make_split(
        df=df,
        y=y,
        split_mode=args.split_mode,
        test_size=args.test_size,
    )

    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    y_train = y.iloc[train_idx].copy()
    y_val = y.iloc[val_idx].copy()

    train_meta = df.iloc[train_idx].copy()
    val_meta = df.iloc[val_idx].copy()

    print("\n[INFO] Training clips:", len(X_train))
    print("[INFO] Validation clips:", len(X_val))

    print("\n[INFO] Training label counts:")
    for label, count in y_train.value_counts().items():
        print(f"  - {label}: {count}")

    print("\n[INFO] Validation label counts:")
    for label, count in y_val.value_counts().items():
        print(f"  - {label}: {count}")

    model = build_model(args.model)

    cv_results = run_cross_validation(model, X, y, df)

    model.fit(X_train, y_train)

    pred = model.predict(X_val)

    labels = sorted(y.unique().tolist())

    acc = accuracy_score(y_val, pred)

    report = classification_report(
        y_val,
        pred,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_val,
        pred,
        labels=labels,
    )

    prob_rows = safe_probabilities(model, X_val, labels)

    prediction_rows = []

    for i, original_idx in enumerate(val_idx):
        meta = val_meta.iloc[i].to_dict()

        row = {
            "clip_group": meta.get("clip_group", ""),
            "session_name": meta.get("session_name", ""),
            "clip_filename": meta.get("clip_filename", ""),
            "true_label": y_val.iloc[i],
            "predicted_label": pred[i],
            "correct": bool(y_val.iloc[i] == pred[i]),
        }

        row.update(prob_rows[i])

        prediction_rows.append(row)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        RESULTS_BASE_DIR
        / f"motion_classifier_{args.model}_{args.split_mode}_{timestamp}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "created_at": datetime.now().isoformat(),
        "features_csv": str(features_csv),
        "model": args.model,
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "accuracy": float(acc),
        "labels": labels,
        "total_clips": int(len(df)),
        "train_clips": int(len(X_train)),
        "val_clips": int(len(X_val)),
        "label_counts": y.value_counts().to_dict(),
        "train_label_counts": y_train.value_counts().to_dict(),
        "val_label_counts": y_val.value_counts().to_dict(),
        "classification_report": report,
        "cross_validation": cv_results,
        "feature_importance": get_feature_importance(model, feature_cols),
        "note": split_note,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(prediction_rows).to_csv(
        run_dir / "predictions.csv",
        index=False,
    )

    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        run_dir / "confusion_matrix.csv",
    )

    train_meta.to_csv(run_dir / "train_split.csv", index=False)
    val_meta.to_csv(run_dir / "val_split.csv", index=False)

    feature_importance = get_feature_importance(model, feature_cols)

    if feature_importance:
        pd.DataFrame(feature_importance).to_csv(
            run_dir / "feature_importance.csv",
            index=False,
        )

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Results saved to: {run_dir}")
    print(f"Validation accuracy: {acc:.4f}")

    if cv_results:
        print("\nCross-validation:")
        for name, result in cv_results.items():
            if isinstance(result, dict) and "mean" in result:
                print(f"  {name}: mean={result['mean']:.4f}, std={result['std']:.4f}")

    print("\nMain files:")
    print(f"  {run_dir / 'metrics.json'}")
    print(f"  {run_dir / 'predictions.csv'}")
    print(f"  {run_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
