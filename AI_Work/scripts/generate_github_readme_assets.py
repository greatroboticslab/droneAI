"""
Generate GitHub README assets for DroneAI results.

Creates:
- accuracy_summary.csv
- accuracy_summary.md
- accuracy_bar.png
- object_flow_examples.png
- time_series_example.png
- class_average_time_series.png
- README_VISUALS_SNIPPET.md

Run from the DroneAI repo root:
    python AI_Work/scripts/generate_github_readme_assets.py

Optional:
    python AI_Work/scripts/generate_github_readme_assets.py \
        --training-runs LabelGUI/TrainingRuns \
        --flow-results LabelGUI/OpticalFlowResults \
        --output AI_Work/results/readme_assets
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


HISTORICAL_RESULTS = [
    {
        "method": "Static ViT baseline",
        "split": "clip",
        "accuracy_pct": 31.58,
        "macro_f1_pct": "",
        "weighted_f1_pct": "",
        "source": "historical",
        "notes": "Static frame baseline; weak because events are motion-based.",
    },
    {
        "method": "Farneback motion RF",
        "split": "clip",
        "accuracy_pct": 42.86,
        "macro_f1_pct": "",
        "weighted_f1_pct": "",
        "source": "historical",
        "notes": "Classical motion features with Random Forest.",
    },
    {
        "method": "Farneback optical-flow LSTM",
        "split": "clip",
        "accuracy_pct": 52.38,
        "macro_f1_pct": "",
        "weighted_f1_pct": "",
        "source": "historical",
        "notes": "Previous optical-flow sequence baseline.",
    },
    {
        "method": "Farneback optical-flow LSTM",
        "split": "session",
        "accuracy_pct": 56.25,
        "macro_f1_pct": "",
        "weighted_f1_pct": "",
        "source": "historical",
        "notes": "Previous optical-flow sequence baseline.",
    },
]


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and value.strip() == "":
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None


def to_pct(value: Any) -> str | float:
    v = safe_float(value)
    if v is None:
        return ""
    if 0 <= v <= 1:
        return round(v * 100, 2)
    return round(v, 2)


def infer_method_from_folder(folder_name: str) -> str:
    name = folder_name.lower()
    if "dpflow" in name and "lstm" in name:
        return "DPFlow-LSTM"
    if "videomae" in name and "finetune" in name:
        return "VideoMAE full fine-tune"
    if "videomae" in name and "probe" in name:
        return "VideoMAE probe"
    if "videomae" in name:
        return "VideoMAE"
    if "farneback" in name and "lstm" in name:
        return "Farneback optical-flow LSTM"
    return folder_name


def infer_split(folder_name: str, metrics: Dict[str, Any]) -> str:
    for key in ["split_mode", "split", "split_type"]:
        if key in metrics and str(metrics[key]).strip():
            return str(metrics[key]).strip()
    low = folder_name.lower()
    if "session" in low:
        return "session"
    if "clip" in low:
        return "clip"
    return ""


def read_metrics(training_runs: Path) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []

    if training_runs.exists():
        for metrics_path in sorted(training_runs.glob("*/metrics.json")):
            run_dir = metrics_path.parent
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception as exc:
                print(f"Could not read {metrics_path}: {exc}")
                continue

            acc = None
            for key in ["best_val_acc", "final_val_acc", "accuracy", "val_accuracy", "val_acc"]:
                if key in metrics:
                    acc = metrics[key]
                    break

            macro = None
            for key in ["macro_f1", "macro avg f1-score", "macro_avg_f1", "best_macro_f1"]:
                if key in metrics:
                    macro = metrics[key]
                    break

            weighted = None
            for key in ["weighted_f1", "weighted avg f1-score", "weighted_avg_f1", "best_weighted_f1"]:
                if key in metrics:
                    weighted = metrics[key]
                    break

            # Some sklearn reports are nested inside metrics.json.
            report = metrics.get("classification_report") or metrics.get("report")
            if isinstance(report, dict):
                if macro is None and isinstance(report.get("macro avg"), dict):
                    macro = report["macro avg"].get("f1-score")
                if weighted is None and isinstance(report.get("weighted avg"), dict):
                    weighted = report["weighted avg"].get("f1-score")

            rows.append({
                "method": infer_method_from_folder(run_dir.name),
                "run_name": run_dir.name,
                "split": infer_split(run_dir.name, metrics),
                "accuracy_pct": to_pct(acc),
                "macro_f1_pct": to_pct(macro),
                "weighted_f1_pct": to_pct(weighted),
                "train_clips": metrics.get("train_clips", ""),
                "val_clips": metrics.get("val_clips", metrics.get("test_clips", "")),
                "best_epoch": metrics.get("best_epoch", ""),
                "source": str(metrics_path),
                "notes": metrics.get("notes", ""),
            })

    hist = pd.DataFrame(HISTORICAL_RESULTS)
    if rows:
        detected = pd.DataFrame(rows)
        # Put historical rows first, then detected experiment rows.
        return pd.concat([hist, detected], ignore_index=True, sort=False)
    return hist


def write_markdown_table(df: pd.DataFrame, out_path: Path) -> str:
    display = df.copy()
    keep_cols = ["method", "split", "accuracy_pct", "macro_f1_pct", "weighted_f1_pct", "run_name", "notes"]
    for col in keep_cols:
        if col not in display.columns:
            display[col] = ""
    display = display[keep_cols]

    headers = ["Method", "Split", "Accuracy", "Macro F1", "Weighted F1", "Run", "Notes"]
    lines = []
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|---|---|---:|---:|---:|---|---|")

    for _, r in display.iterrows():
        def fmt_pct(x: Any) -> str:
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                return f"{float(x):.2f}%"
            if isinstance(x, str) and x.strip():
                try:
                    return f"{float(x):.2f}%"
                except Exception:
                    return x
            return ""

        line = [
            str(r.get("method", "")),
            str(r.get("split", "")),
            fmt_pct(r.get("accuracy_pct", "")),
            fmt_pct(r.get("macro_f1_pct", "")),
            fmt_pct(r.get("weighted_f1_pct", "")),
            str(r.get("run_name", "")),
            str(r.get("notes", "")),
        ]
        line = [v.replace("|", "\\|") for v in line]
        lines.append("| " + " | ".join(line) + " |")

    md = "\n".join(lines) + "\n"
    out_path.write_text(md, encoding="utf-8")
    return md


def make_accuracy_chart(df: pd.DataFrame, out_path: Path) -> None:
    plot_df = df.copy()
    plot_df["accuracy_num"] = pd.to_numeric(plot_df["accuracy_pct"], errors="coerce")
    plot_df = plot_df.dropna(subset=["accuracy_num"])

    # Keep the chart readable. Prefer high-level rows and latest detected runs.
    if len(plot_df) > 12:
        plot_df = plot_df.tail(12)

    labels = []
    for _, r in plot_df.iterrows():
        method = str(r.get("method", ""))
        split = str(r.get("split", ""))
        labels.append(f"{method}\n({split})" if split else method)

    fig, ax = plt.subplots(figsize=(max(9, len(plot_df) * 1.05), 5.4))
    ax.bar(range(len(plot_df)), plot_df["accuracy_num"].values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("DroneAI event-classification accuracy")
    ax.set_xticks(range(len(plot_df)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.25)

    for i, val in enumerate(plot_df["accuracy_num"].values):
        ax.text(i, val + 1.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def find_best_flow_run(flow_results: Path) -> Optional[Path]:
    candidates: List[Tuple[float, Path]] = []
    for summary_path in flow_results.glob("*/run_summary.json"):
        try:
            data = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rate = safe_float(data.get("mean_roi_available_rate"))
        if rate is not None:
            candidates.append((rate, summary_path.parent))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def copy_object_flow_examples(flow_run: Path, output: Path, max_images: int = 3) -> List[Path]:
    debug_dir = flow_run / "debug_flow_images"
    if not debug_dir.exists():
        return []
    images = sorted([p for p in debug_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])
    copied = []
    for i, p in enumerate(images[:max_images], start=1):
        dest = output / f"object_flow_example_{i:02d}{p.suffix.lower()}"
        shutil.copy2(p, dest)
        copied.append(dest)
    return copied


def make_contact_sheet(image_paths: List[Path], out_path: Path) -> None:
    if not image_paths:
        return
    import cv2

    imgs = []
    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        target_w = 1200
        scale = target_w / w
        resized = cv2.resize(img, (target_w, int(h * scale)))
        imgs.append(resized)

    if not imgs:
        return

    widths = [im.shape[1] for im in imgs]
    max_w = max(widths)
    padded = []
    for im in imgs:
        if im.shape[1] < max_w:
            pad = np.zeros((im.shape[0], max_w - im.shape[1], 3), dtype=np.uint8)
            im = np.hstack([im, pad])
        padded.append(im)

    sheet = np.vstack(padded)
    # Use matplotlib to save RGB image without managing BGR conversion.
    plt.imsave(out_path, sheet)


def pick_representative_clip(seq_df: pd.DataFrame) -> str:
    tmp = seq_df.copy()
    tmp["roi_available"] = pd.to_numeric(tmp.get("roi_available", 0), errors="coerce").fillna(0)
    tmp["both_detected"] = pd.to_numeric(tmp.get("both_detected", 0), errors="coerce").fillna(0)
    tmp["score"] = tmp["roi_available"] + tmp["both_detected"]
    grp = tmp.groupby("clip_group")["score"].mean().sort_values(ascending=False)
    if len(grp):
        return str(grp.index[0])
    return str(seq_df["clip_group"].iloc[0])


def make_time_series_plots(flow_run: Path, output: Path) -> Tuple[Optional[Path], Optional[Path]]:
    seq_path = flow_run / "flow_sequence_features.csv"
    if not seq_path.exists():
        return None, None

    df = pd.read_csv(seq_path)
    if df.empty or "clip_group" not in df.columns:
        return None, None

    for col in ["step_index", "flow_mag_norm_per_sec", "det_speed_norm_per_sec", "det_vy_norm_per_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Plot one representative clip.
    clip = pick_representative_clip(df)
    clip_df = df[df["clip_group"] == clip].sort_values("step_index")
    single_out = output / "time_series_example.png"

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = clip_df["step_index"].values
    if "flow_mag_norm_per_sec" in clip_df.columns:
        ax.plot(x, clip_df["flow_mag_norm_per_sec"].values, marker="o", label="DPFlow magnitude")
    if "det_speed_norm_per_sec" in clip_df.columns:
        ax.plot(x, clip_df["det_speed_norm_per_sec"].values, marker="o", label="Detected-center speed")
    if "det_vy_norm_per_sec" in clip_df.columns:
        ax.plot(x, clip_df["det_vy_norm_per_sec"].values, marker="o", label="Vertical velocity")
    ax.set_title("Example object-centered motion time series")
    ax.set_xlabel("Frame-pair step")
    ax.set_ylabel("Normalized motion per second")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(single_out, dpi=180)
    plt.close(fig)

    # Class-average time series with normalized clip time bins.
    avg_out = output / "class_average_time_series.png"
    if "label" in df.columns and "flow_mag_norm_per_sec" in df.columns:
        rows = []
        for clip_group, g in df.groupby("clip_group"):
            g = g.sort_values("step_index").copy()
            label = str(g["label"].iloc[0])
            n = len(g)
            if n == 0:
                continue
            g["time_bin"] = np.floor(np.linspace(0, 19, n)).astype(int)
            for b, gb in g.groupby("time_bin"):
                rows.append({
                    "label": label,
                    "time_bin": int(b),
                    "flow_mag_norm_per_sec": float(pd.to_numeric(gb["flow_mag_norm_per_sec"], errors="coerce").mean()),
                })
        avg = pd.DataFrame(rows)
        if not avg.empty:
            avg = avg.groupby(["label", "time_bin"], as_index=False)["flow_mag_norm_per_sec"].mean()
            fig, ax = plt.subplots(figsize=(9, 4.8))
            for label, g in avg.groupby("label"):
                g = g.sort_values("time_bin")
                ax.plot(g["time_bin"].values, g["flow_mag_norm_per_sec"].values, marker="o", label=label)
            ax.set_title("Class-average DPFlow time-series pattern")
            ax.set_xlabel("Normalized clip time bin")
            ax.set_ylabel("Mean normalized flow magnitude per second")
            ax.grid(alpha=0.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(avg_out, dpi=180)
            plt.close(fig)
            return single_out, avg_out

    return single_out, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-runs", default="LabelGUI/TrainingRuns")
    parser.add_argument("--flow-results", default="LabelGUI/OpticalFlowResults")
    parser.add_argument("--flow-run", default="", help="Optional exact flow run folder. If omitted, the script chooses the run with the best ROI availability.")
    parser.add_argument("--output", default="AI_Work/results/readme_assets")
    args = parser.parse_args()

    training_runs = Path(args.training_runs)
    flow_results = Path(args.flow_results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    df = read_metrics(training_runs)
    csv_path = output / "accuracy_summary.csv"
    md_path = output / "accuracy_summary.md"
    chart_path = output / "accuracy_bar.png"

    df.to_csv(csv_path, index=False)
    table_md = write_markdown_table(df, md_path)
    make_accuracy_chart(df, chart_path)

    if args.flow_run:
        flow_run = Path(args.flow_run)
    else:
        flow_run = find_best_flow_run(flow_results)

    object_images: List[Path] = []
    object_sheet = None
    ts_one = None
    ts_avg = None
    flow_summary = {}

    if flow_run is not None and flow_run.exists():
        summary_path = flow_run / "run_summary.json"
        if summary_path.exists():
            try:
                flow_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                flow_summary = {}
        object_images = copy_object_flow_examples(flow_run, output, max_images=3)
        object_sheet = output / "object_flow_examples.png"
        make_contact_sheet(object_images, object_sheet)
        ts_one, ts_avg = make_time_series_plots(flow_run, output)

    # Build README snippet with relative links from repo root.
    rel = lambda p: str(p).replace("\\", "/")
    snippet = []
    snippet.append("## DroneAI Progress Visualizations\n")
    snippet.append("### Accuracy Table\n")
    snippet.append("The table below tracks model accuracy and keeps earlier baseline results for paper writing.\n")
    snippet.append(table_md)
    snippet.append("\n![DroneAI event-classification accuracy](AI_Work/results/readme_assets/accuracy_bar.png)\n")

    snippet.append("\n### Object-Centered Optical Flow Visualization\n")
    if flow_summary:
        roi = safe_float(flow_summary.get("mean_roi_available_rate"))
        both = safe_float(flow_summary.get("mean_both_detected_rate"))
        if roi is not None and both is not None:
            snippet.append(f"The corrected DPFlow extraction achieved ROI availability of **{roi*100:.2f}%** and both-frame detection availability of **{both*100:.2f}%**.\n")
    if object_sheet and object_sheet.exists():
        snippet.append("\n![Object-centered optical flow examples](AI_Work/results/readme_assets/object_flow_examples.png)\n")
    elif object_images:
        snippet.append("\n![Object-centered optical flow example](AI_Work/results/readme_assets/object_flow_example_01.jpg)\n")
    else:
        snippet.append("\nObject-flow examples were not found. Run DPFlow extraction with debug images enabled.\n")

    snippet.append("\n### Time-Series Motion Data\n")
    snippet.append("These plots show the motion signal over time instead of treating each frame independently. This is important because takeoff, landing, and crash are motion events.\n")
    if ts_one and ts_one.exists():
        snippet.append("\n![Example motion time series](AI_Work/results/readme_assets/time_series_example.png)\n")
    if ts_avg and ts_avg.exists():
        snippet.append("\n![Class-average motion time series](AI_Work/results/readme_assets/class_average_time_series.png)\n")

    snippet_path = output / "README_VISUALS_SNIPPET.md"
    snippet_path.write_text("".join(snippet), encoding="utf-8")

    print("Created README assets in:", output.resolve())
    print("-", csv_path)
    print("-", md_path)
    print("-", chart_path)
    print("-", snippet_path)
    if flow_run:
        print("Selected flow run:", flow_run)
    if object_sheet and object_sheet.exists():
        print("-", object_sheet)
    if ts_one and ts_one.exists():
        print("-", ts_one)
    if ts_avg and ts_avg.exists():
        print("-", ts_avg)


if __name__ == "__main__":
    main()
