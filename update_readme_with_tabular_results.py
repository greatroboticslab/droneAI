from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
README = ROOT / "AI_Work" / "README.md"
ASSET_DIR = ROOT / "AI_Work" / "results" / "readme_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)

BASE_ROWS = [
    ["Static ViT baseline", "clip", 31.58, None, None, "Static-frame baseline. Weak because events are motion-based."],
    ["Farneback motion RF", "clip", 42.86, None, None, "Classical motion features with Random Forest."],
    ["Farneback optical-flow LSTM", "clip", 52.38, None, None, "Previous optical-flow sequence baseline."],
    ["Farneback optical-flow LSTM", "session", 56.25, None, None, "Previous optical-flow sequence baseline."],
    ["DPFlow optical-flow LSTM", "clip", 52.38, 52.38, 52.72, "LSTM trained on corrected DPFlow motion sequence features."],
    ["DPFlow optical-flow LSTM", "session", 62.50, 54.19, 60.73, "Previous best accuracy result."],
    ["VideoMAE probe", "clip", 42.86, 35.38, 40.43, "Frozen pretrained VideoMAE backbone; trained only classification head."],
    ["VideoMAE probe", "session", 37.50, 35.00, 35.63, "Frozen pretrained VideoMAE backbone."],
    ["VideoMAE full fine-tune", "clip", 52.38, 51.50, 51.71, "Full VideoMAE fine-tuning improved over the frozen probe."],
    ["VideoMAE full fine-tune", "session", 56.25, 53.33, 57.08, "Full VideoMAE fine-tuning, but below DPFlow-LSTM."],
]

TABULAR_RUNS = [
    ROOT / "LabelGUI" / "TrainingRuns" / "dpflow_tabular_clip_v1",
    ROOT / "LabelGUI" / "TrainingRuns" / "dpflow_tabular_session_v1",
]

def pretty_model_name(name):
    return str(name).replace("_", " ").title()

def pct_value(x):
    if x is None or x == "":
        return None
    x = float(x)
    if x <= 1.0:
        x *= 100.0
    return x

def read_tabular_rows():
    rows = []
    for run_dir in TABULAR_RUNS:
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            print("Missing:", metrics_path)
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            m = json.load(f)

        best_model = pretty_model_name(m.get("best_model", "unknown"))
        split = m.get("split_mode", run_dir.name)

        rows.append([
            f"DPFlow tabular {best_model}",
            split,
            pct_value(m.get("accuracy")),
            pct_value(m.get("macro_f1")),
            pct_value(m.get("weighted_f1")),
            "Feature-engineered DPFlow summary statistics with a tree-based tabular model.",
        ])

    return rows

def fmt(x):
    if x is None:
        return "—"
    return f"{float(x):.2f}%"

def make_accuracy_files(rows):
    df = pd.DataFrame(rows, columns=["Method", "Split", "Accuracy", "Macro F1", "Weighted F1", "Notes"])
    df.to_csv(ASSET_DIR / "accuracy_summary.csv", index=False)

    with open(ASSET_DIR / "accuracy_summary.md", "w", encoding="utf-8") as f:
        f.write("| Method | Split | Accuracy | Macro F1 | Weighted F1 | Notes |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for method, split, acc, macro, weighted, notes in rows:
            f.write(f"| {method} | {split} | {fmt(acc)} | {fmt(macro)} | {fmt(weighted)} | {notes} |\n")

    labels = [f"{r[0]}\n({r[1]})" for r in rows]
    values = [r[2] for r in rows]

    plt.figure(figsize=(13, 8))
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels, fontsize=8)
    plt.xlabel("Validation accuracy (%)")
    plt.title("DroneAI Event-Classification Accuracy Across Experiments")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.25)

    for i, v in enumerate(values):
        plt.text(v + 1, i, f"{v:.2f}%", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(ASSET_DIR / "accuracy_bar.png", dpi=180)
    plt.close()

def make_feature_importance_plot(run_dir, output_name, title):
    out_path = ASSET_DIR / output_name
    p = run_dir / "feature_importance.csv"

    if not p.exists():
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, f"Feature importance file not found:\n{p}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    df = pd.read_csv(p)
    if df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "Feature importance file is empty.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    cols = {c.lower(): c for c in df.columns}

    feature_col = cols.get("feature") or cols.get("feature_name") or df.columns[0]
    importance_col = cols.get("importance") or cols.get("feature_importance") or cols.get("value") or df.columns[-1]

    plot_df = df[[feature_col, importance_col]].copy()
    plot_df[importance_col] = pd.to_numeric(plot_df[importance_col], errors="coerce")
    plot_df = plot_df.dropna().sort_values(importance_col, ascending=False).head(15)

    if plot_df.empty:
        plt.figure(figsize=(10, 4))
        plt.text(0.5, 0.5, "No numeric feature importance values found.", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    plot_df = plot_df.iloc[::-1]

    plt.figure(figsize=(11, 6))
    plt.barh(plot_df[feature_col], plot_df[importance_col])
    plt.xlabel("Feature importance")
    plt.title(title)
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()

def update_readme(rows):
    if not README.exists():
        raise FileNotFoundError(f"Could not find {README}")

    text = README.read_text(encoding="utf-8")
    marker = "## DroneAI Progress Visualizations"

    if marker in text:
        text = text.split(marker)[0].rstrip()

    if text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```\n"

    table_lines = [
        "| Method | Split | Accuracy | Macro F1 | Weighted F1 | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for method, split, acc, macro, weighted, notes in rows:
        bold = method.startswith("DPFlow tabular") or method == "DPFlow optical-flow LSTM"
        if bold:
            table_lines.append(
                f"| **{method}** | **{split}** | **{fmt(acc)}** | **{fmt(macro)}** | **{fmt(weighted)}** | {notes} |"
            )
        else:
            table_lines.append(
                f"| {method} | {split} | {fmt(acc)} | {fmt(macro)} | {fmt(weighted)} | {notes} |"
            )

    table_md = "\n".join(table_lines)

    section = f"""
## DroneAI Progress Visualizations

This section tracks the current experimental progress for DroneAI. The goal is not only to show the best number, but also to show what was tried, what improved, and what still needs work.

The main direction of the project has shifted from static image classification toward motion-based video understanding. This is because takeoff, landing, and crash events are not single-frame events. They depend on how the drone moves across time.

### Accuracy Table

The newest experiment is the **DPFlow tabular baseline**. Instead of giving the LSTM the raw motion sequence, this approach summarizes each clip into physical motion statistics such as peak speed, average speed, vertical motion, acceleration-style changes, motion variation, and optical-flow energy.

This is useful because the current dataset is still small. Tree-based tabular models can sometimes work better than LSTMs when there are not many clips.

{table_md}

![DroneAI event-classification accuracy](results/readme_assets/accuracy_bar.png)

### New Result: DPFlow Tabular Features

The DPFlow tabular model improved the clip split result from **52.38%** with DPFlow-LSTM to **57.14%** with Random Forest.

On the session split, the tabular model matched the DPFlow-LSTM accuracy at **62.50%**, but improved the F1 scores:

| Split | Previous DPFlow-LSTM | New DPFlow tabular model | Interpretation |
|---|---:|---:|---|
| Clip accuracy | 52.38% | **57.14%** | Tabular motion features improved the clip split. |
| Session accuracy | 62.50% | **62.50%** | Accuracy tied the previous best. |
| Session macro F1 | 54.19% | **60.00%** | Better class balance than the LSTM. |
| Session weighted F1 | 60.73% | **63.13%** | Better overall F1 than the LSTM. |

This supports the idea that with only 83 clips, feature-engineered motion statistics may be more reliable than raw sequence learning.

### Feature Importance

The feature-importance plots help show which motion statistics the tree-based models used most. This is useful for the research paper because it makes the model easier to explain compared to a black-box LSTM.

![DPFlow tabular feature importance, clip split](results/readme_assets/tabular_feature_importance_clip.png)

![DPFlow tabular feature importance, session split](results/readme_assets/tabular_feature_importance_session.png)

### Object-Centered Optical Flow Visualization

The optical-flow pipeline uses the drone detector to focus motion estimation around the drone instead of treating the entire frame equally. This matters because the camera, floor, background, and lighting can introduce motion noise.

In the corrected DPFlow extraction run, the drone ROI was available for about **89.47%** of frame-to-frame motion steps, and both consecutive frames had detections for about **78.19%** of steps.

![Object-centered DPFlow optical-flow examples](results/readme_assets/object_flow_examples.png)

### Time-Series Motion Data

Each labeled clip is treated as a short motion sequence. The time-series plots show how motion features change across the clip. This is important because the difference between takeoff, landing, minor crash, and severe crash often appears in the motion pattern over time.

![Example time-series motion data](results/readme_assets/time_series_example.png)

![Class-average time-series motion data](results/readme_assets/class_average_time_series.png)

### Current Interpretation

The best current accuracy is still **62.50%**, but the new DPFlow tabular result is important because it matched the best LSTM accuracy while improving F1. It also gives feature importance, which helps explain what physical motion signals are actually useful.

The main limitation is still dataset size. The current dataset has only 83 clips, and the landing class is especially small. The next major step is to expand the labeled dataset and continue testing models that combine motion features with visual video features.
"""

    README.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")

def main():
    tabular_rows = read_tabular_rows()
    rows = BASE_ROWS + tabular_rows

    make_accuracy_files(rows)

    make_feature_importance_plot(
        ROOT / "LabelGUI" / "TrainingRuns" / "dpflow_tabular_clip_v1",
        "tabular_feature_importance_clip.png",
        "DPFlow Tabular Feature Importance: Clip Split",
    )

    make_feature_importance_plot(
        ROOT / "LabelGUI" / "TrainingRuns" / "dpflow_tabular_session_v1",
        "tabular_feature_importance_session.png",
        "DPFlow Tabular Feature Importance: Session Split",
    )

    update_readme(rows)

    print("Updated README and assets:")
    print(" -", README)
    print(" -", ASSET_DIR / "accuracy_bar.png")
    print(" -", ASSET_DIR / "tabular_feature_importance_clip.png")
    print(" -", ASSET_DIR / "tabular_feature_importance_session.png")

if __name__ == "__main__":
    main()
