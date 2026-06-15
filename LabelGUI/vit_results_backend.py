import json
import re
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
VIT_RESULTS_DIR = BASE_DIR / "ViTResults"
FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._ ")
    return name


def _is_safe_child(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def _load_json(path: Path):
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def list_vit_runs():
    """
    Lists folders inside LabelGUI/ViTResults.
    Newest runs appear first.
    """
    VIT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    runs = []

    for folder in sorted(VIT_RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not folder.is_dir():
            continue

        metrics_path = folder / "metrics.json"
        metrics = _load_json(metrics_path)

        runs.append({
            "name": folder.name,
            "path": str(folder),
            "created_at": metrics.get("created_at", ""),
            "final_val_acc": metrics.get("final_val_acc", None),
            "total_images": metrics.get("total_images", None),
            "train_images": metrics.get("train_images", None),
            "val_images": metrics.get("val_images", None),
        })

    return runs


def load_vit_run(run_name: str, view_filter: str = "all", limit: int = 60):
    """
    Loads one ViT run folder and prepares data for the GUI.
    """
    run_name = _safe_name(run_name)

    run_dir = VIT_RESULTS_DIR / run_name

    if not _is_safe_child(VIT_RESULTS_DIR, run_dir):
        raise ValueError("Invalid ViT results folder.")

    if not run_dir.exists():
        raise FileNotFoundError(f"ViT run not found: {run_name}")

    metrics = _load_json(run_dir / "metrics.json")
    labels = _load_json(run_dir / "labels.json")

    # -----------------------------
    # Label counts
    # -----------------------------
    label_counts = metrics.get("label_counts", {})
    label_count_rows = [
        {"label": label, "count": count}
        for label, count in sorted(label_counts.items())
    ]

    # -----------------------------
    # Training history
    # -----------------------------
    history = metrics.get("history", [])

    # -----------------------------
    # Confusion matrix
    # -----------------------------
    confusion = {
        "columns": [],
        "rows": [],
    }

    cm_path = run_dir / "confusion_matrix.csv"
    if cm_path.exists():
        try:
            cm_df = pd.read_csv(cm_path, index_col=0)
            confusion["columns"] = cm_df.columns.tolist()

            for idx, row in cm_df.iterrows():
                confusion["rows"].append({
                    "label": idx,
                    "values": [int(v) for v in row.tolist()],
                })
        except Exception:
            pass

    # -----------------------------
    # Prediction samples
    # -----------------------------
    predictions = []

    pred_path = run_dir / "prediction_samples.csv"
    if pred_path.exists():
        try:
            pred_df = pd.read_csv(pred_path)

            if view_filter == "correct":
                pred_df = pred_df[pred_df["correct"] == True]
            elif view_filter == "wrong":
                pred_df = pred_df[pred_df["correct"] == False]

            pred_df = pred_df.head(limit)

            for _, row in pred_df.iterrows():
                predictions.append({
                    "image_path": str(row.get("image_path", "")).replace("\\", "/"),
                    "true_label": row.get("true_label", ""),
                    "predicted_label": row.get("predicted_label", ""),
                    "confidence": row.get("confidence", ""),
                    "correct": bool(row.get("correct", False)),
                })
        except Exception:
            pass

    summary = {
        "run_name": run_name,
        "created_at": metrics.get("created_at", ""),
        "device": metrics.get("device", ""),
        "total_images": metrics.get("total_images", 0),
        "train_images": metrics.get("train_images", 0),
        "val_images": metrics.get("val_images", 0),
        "final_val_acc": metrics.get("final_val_acc", None),
        "final_val_loss": metrics.get("final_val_loss", None),
        "note": metrics.get("note", ""),
    }

    return {
        "summary": summary,
        "labels": labels,
        "label_counts": label_count_rows,
        "history": history,
        "confusion": confusion,
        "predictions": predictions,
        "view_filter": view_filter,
    }
