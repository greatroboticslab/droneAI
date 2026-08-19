from pathlib import Path
import math
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


ROOT = Path(".")
README = ROOT / "AI_Work" / "README.md"
ASSET_DIR = ROOT / "AI_Work" / "results" / "readme_assets"
ASSET_DIR.mkdir(parents=True, exist_ok=True)


accuracy_rows = [
    {
        "Method": "Static ViT baseline",
        "Split": "clip",
        "Accuracy": 31.58,
        "Macro F1": "",
        "Weighted F1": "",
        "Notes": "Static-frame baseline. This was weak because the event labels depend on motion, not just one image."
    },
    {
        "Method": "Farneback motion RF",
        "Split": "clip",
        "Accuracy": 42.86,
        "Macro F1": "",
        "Weighted F1": "",
        "Notes": "Classical motion features with Random Forest."
    },
    {
        "Method": "Farneback optical-flow LSTM",
        "Split": "clip",
        "Accuracy": 52.38,
        "Macro F1": "",
        "Weighted F1": "",
        "Notes": "Earlier optical-flow sequence baseline."
    },
    {
        "Method": "Farneback optical-flow LSTM",
        "Split": "session",
        "Accuracy": 56.25,
        "Macro F1": "",
        "Weighted F1": "",
        "Notes": "Earlier optical-flow sequence baseline."
    },
    {
        "Method": "DPFlow optical-flow LSTM",
        "Split": "clip",
        "Accuracy": 52.38,
        "Macro F1": 52.38,
        "Weighted F1": 52.72,
        "Notes": "Uses newer DPFlow optical-flow features. ROI bug fixed."
    },
    {
        "Method": "DPFlow optical-flow LSTM",
        "Split": "session",
        "Accuracy": 62.50,
        "Macro F1": 54.19,
        "Weighted F1": 60.73,
        "Notes": "Current best result."
    },
    {
        "Method": "VideoMAE probe",
        "Split": "clip",
        "Accuracy": 42.86,
        "Macro F1": 35.38,
        "Weighted F1": 40.43,
        "Notes": "Frozen pretrained VideoMAE backbone; trained only classification head."
    },
    {
        "Method": "VideoMAE probe",
        "Split": "session",
        "Accuracy": 37.50,
        "Macro F1": 35.00,
        "Weighted F1": 35.63,
        "Notes": "Frozen pretrained VideoMAE backbone; trained only classification head."
    },
    {
        "Method": "VideoMAE full fine-tune",
        "Split": "clip",
        "Accuracy": 52.38,
        "Macro F1": 51.50,
        "Weighted F1": 51.71,
        "Notes": "Full VideoMAE fine-tuning improved over probe but did not beat DPFlow-LSTM."
    },
    {
        "Method": "VideoMAE full fine-tune",
        "Split": "session",
        "Accuracy": 56.25,
        "Macro F1": 53.33,
        "Weighted F1": 57.08,
        "Notes": "Full VideoMAE fine-tuning."
    },
]


def make_accuracy_outputs():
    df = pd.DataFrame(accuracy_rows)

    df.to_csv(ASSET_DIR / "accuracy_summary.csv", index=False)

    with open(ASSET_DIR / "accuracy_summary.md", "w", encoding="utf-8") as f:
        f.write("| Method | Split | Accuracy | Macro F1 | Weighted F1 | Notes |\n")
        f.write("|---|---:|---:|---:|---:|---|\n")
        for r in accuracy_rows:
            f.write(
                f"| {r['Method']} | {r['Split']} | {r['Accuracy']:.2f}% | "
                f"{r['Macro F1'] if r['Macro F1'] != '' else ''} | "
                f"{r['Weighted F1'] if r['Weighted F1'] != '' else ''} | "
                f"{r['Notes']} |\n"
            )

    labels = [f"{r['Method']}\n({r['Split']})" for r in accuracy_rows]
    values = [r["Accuracy"] for r in accuracy_rows]

    plt.figure(figsize=(12, 7))
    y = np.arange(len(labels))
    plt.barh(y, values)
    plt.yticks(y, labels)
    plt.xlabel("Validation accuracy (%)")
    plt.title("DroneAI Event-Classification Accuracy Across Experiments")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.25)

    for i, v in enumerate(values):
        plt.text(v + 1, i, f"{v:.2f}%", va="center")

    plt.tight_layout()
    plt.savefig(ASSET_DIR / "accuracy_bar.png", dpi=180)
    plt.close()


def find_flow_run():
    candidates = [
        ROOT / "LabelGUI" / "OpticalFlowResults" / "dpflow_drone_v2_gpu_keyfix",
        ROOT / "LabelGUI" / "OpticalFlowResults" / "dpflow_drone_v1_gpu",
        ROOT / "LabelGUI" / "OpticalFlowResults" / "flow_v1",
    ]

    for c in candidates:
        if (c / "flow_sequence_features.csv").exists():
            return c

    found = list((ROOT / "LabelGUI" / "OpticalFlowResults").glob("*/flow_sequence_features.csv"))
    if found:
        return found[0].parent

    return None


def make_object_flow_collage(flow_run):
    out_path = ASSET_DIR / "object_flow_examples.png"

    image_paths = []
    if flow_run is not None:
        image_paths = sorted((flow_run / "debug_flow_images").glob("*.jpg"))
        image_paths += sorted((flow_run / "debug_flow_images").glob("*.png"))

    if not image_paths:
        plt.figure(figsize=(12, 4))
        plt.text(
            0.5,
            0.5,
            "Object-centered optical-flow debug images were not found locally.\n"
            "Copy the debug_flow_images folder from the DPFlow extraction run and regenerate this section.",
            ha="center",
            va="center",
            fontsize=13,
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        return

    selected = image_paths[:3]

    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 4 * len(selected)))
    if len(selected) == 1:
        axes = [axes]

    for ax, p in zip(axes, selected):
        img = mpimg.imread(p)
        ax.imshow(img)
        ax.set_title(p.name, fontsize=10)
        ax.axis("off")

    plt.suptitle("Object-Centered DPFlow Optical-Flow Examples", fontsize=15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def make_time_series_plots(flow_run):
    single_out = ASSET_DIR / "time_series_example.png"
    class_out = ASSET_DIR / "class_average_time_series.png"

    if flow_run is None or not (flow_run / "flow_sequence_features.csv").exists():
        for out_path in [single_out, class_out]:
            plt.figure(figsize=(11, 4))
            plt.text(
                0.5,
                0.5,
                "DPFlow time-series CSV was not found locally.\n"
                "Copy flow_sequence_features.csv from the DPFlow extraction run and regenerate.",
                ha="center",
                va="center",
                fontsize=13,
            )
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(out_path, dpi=180)
            plt.close()
        return

    df = pd.read_csv(flow_run / "flow_sequence_features.csv")

    needed = ["clip_group", "label", "step_index"]
    for col in needed:
        if col not in df.columns:
            raise ValueError(f"Missing required column in flow_sequence_features.csv: {col}")

    for col in ["flow_mag_norm_per_sec", "det_speed_norm_per_sec", "det_vy_norm_per_sec"]:
        if col not in df.columns:
            df[col] = 0.0

    df = df.sort_values(["clip_group", "step_index"]).copy()

    # Pick a useful example clip with several steps and nonzero movement.
    grouped = []
    for clip_group, g in df.groupby("clip_group"):
        movement = float(g["det_speed_norm_per_sec"].fillna(0).mean())
        grouped.append((clip_group, len(g), movement, g["label"].iloc[0]))

    grouped = sorted(grouped, key=lambda x: (x[1], x[2]), reverse=True)

    if grouped:
        selected_clip = grouped[0][0]
        g = df[df["clip_group"] == selected_clip].sort_values("step_index")
    else:
        g = df.head(0)

    plt.figure(figsize=(11, 5))
    if len(g):
        x = g["step_index"].to_numpy()
        plt.plot(x, g["flow_mag_norm_per_sec"].fillna(0), marker="o", label="DPFlow magnitude")
        plt.plot(x, g["det_speed_norm_per_sec"].fillna(0), marker="o", label="Detected-center speed")
        plt.plot(x, g["det_vy_norm_per_sec"].fillna(0), marker="o", label="Vertical motion")
        plt.title(f"Example Time-Series Motion Signal: {g['label'].iloc[0]}")
        plt.xlabel("Frame-to-frame step")
        plt.ylabel("Normalized motion per second")
        plt.legend()
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "No time-series rows found.", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(single_out, dpi=180)
    plt.close()

    # Class-average normalized time series.
    rows = []
    for clip_group, g in df.groupby("clip_group"):
        g = g.sort_values("step_index").copy()
        if len(g) < 2:
            continue

        label = g["label"].iloc[0]
        max_step = max(float(g["step_index"].max()), 1.0)

        for _, row in g.iterrows():
            pct = int(round((float(row["step_index"]) / max_step) * 10)) * 10
            rows.append({
                "label": label,
                "pct": pct,
                "det_speed_norm_per_sec": float(row.get("det_speed_norm_per_sec", 0) or 0),
                "flow_mag_norm_per_sec": float(row.get("flow_mag_norm_per_sec", 0) or 0),
            })

    avg = pd.DataFrame(rows)

    plt.figure(figsize=(11, 5))
    if len(avg):
        for label, g in avg.groupby("label"):
            curve = g.groupby("pct")["det_speed_norm_per_sec"].mean().reset_index()
            plt.plot(curve["pct"], curve["det_speed_norm_per_sec"], marker="o", label=label)

        plt.title("Class-Average Detected-Center Speed Over Normalized Clip Time")
        plt.xlabel("Normalized clip time (%)")
        plt.ylabel("Average normalized speed per second")
        plt.legend()
        plt.grid(alpha=0.25)
    else:
        plt.text(0.5, 0.5, "No class-average time-series data found.", ha="center", va="center")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(class_out, dpi=180)
    plt.close()


def update_readme():
    if not README.exists():
        raise FileNotFoundError(f"Could not find {README}")

    text = README.read_text(encoding="utf-8")

    marker = "## DroneAI Progress Visualizations"
    if marker in text:
        text = text.split(marker)[0].rstrip()

    # If the old append happened inside an open Markdown code block,
    # close that code block before adding the new rendered markdown section.
    if text.count("```") % 2 == 1:
        text = text.rstrip() + "\n```\n"

    section = r'''
## DroneAI Progress Visualizations

This section tracks the current experimental progress for DroneAI. The goal is not just to report the best number, but to keep a clear record of what was tried, what improved, and what still needs work. This is important for the final paper because some experiments were useful even when they did not produce the highest accuracy.

At this stage, the project has moved from static image classification toward motion-based video understanding. The main finding so far is that drone events are difficult to classify from single frames alone. Takeoff, landing, and crash events depend on how the drone moves over time, so the most useful features are speed, direction, acceleration, optical flow, and temporal patterns across the clip.

### Accuracy Table

The table below keeps the main model results in one place. The best result so far is the DPFlow optical-flow LSTM on the session split. The dataset is still small, so these results should be treated as progress indicators rather than final benchmark numbers.

| Method | Split | Accuracy | Macro F1 | Weighted F1 | Notes |
|---|---:|---:|---:|---:|---|
| Static ViT baseline | clip | 31.58% |  |  | Static-frame baseline. Weak because the events are motion-based. |
| Farneback motion RF | clip | 42.86% |  |  | Classical motion features with Random Forest. |
| Farneback optical-flow LSTM | clip | 52.38% |  |  | Previous optical-flow sequence baseline. |
| Farneback optical-flow LSTM | session | 56.25% |  |  | Previous optical-flow sequence baseline. |
| DPFlow optical-flow LSTM | clip | 52.38% | 52.38% | 52.72% | Newer optical-flow method with fixed drone ROI matching. |
| **DPFlow optical-flow LSTM** | **session** | **62.50%** | **54.19%** | **60.73%** | **Current best result.** |
| VideoMAE probe | clip | 42.86% | 35.38% | 40.43% | Frozen pretrained VideoMAE backbone; trained only the classification head. |
| VideoMAE probe | session | 37.50% | 35.00% | 35.63% | Frozen pretrained VideoMAE backbone. |
| VideoMAE full fine-tune | clip | 52.38% | 51.50% | 51.71% | Full VideoMAE fine-tuning improved over the frozen probe. |
| VideoMAE full fine-tune | session | 56.25% | 53.33% | 57.08% | Full VideoMAE fine-tuning, but still below DPFlow-LSTM. |

![DroneAI event-classification accuracy](results/readme_assets/accuracy_bar.png)

### Object-Centered Optical Flow Visualization

The optical-flow pipeline uses the drone detector to focus motion estimation around the drone instead of treating the entire frame equally. This matters because the camera, floor, background, and lighting can introduce motion noise. The debug images below are included to make the method easier to inspect visually.

In the DPFlow extraction run, the drone ROI was available for about **89.47%** of frame-to-frame motion steps, and both consecutive frames had detections for about **78.19%** of steps. This shows that the detector and optical-flow pipeline are connected correctly after fixing the earlier path-matching issue.

![Object-centered DPFlow optical-flow examples](results/readme_assets/object_flow_examples.png)

### Time-Series Motion Data

Each labeled clip is treated as a short motion sequence, not as a group of unrelated frames. The time-series plots show how motion features change across the clip. This is important because the difference between takeoff, landing, minor crash, and severe crash often appears in the motion pattern over time.

The current time-series features include optical-flow magnitude, detected-center speed, vertical motion, and acceleration-style changes. These features are used by the LSTM classifier to learn event-level motion patterns.

![Example time-series motion data](results/readme_assets/time_series_example.png)

![Class-average time-series motion data](results/readme_assets/class_average_time_series.png)

### Current Interpretation

The results show that stronger optical flow helps, but optical flow alone is not enough to fully solve the task with only 83 clips. DPFlow produced the best current result, but the model is still limited by dataset size and class imbalance, especially for landing clips. The next major step is to expand the labeled dataset and continue testing models that combine visual video features with motion-based features.
'''

    README.write_text(text.rstrip() + "\n\n" + section.strip() + "\n", encoding="utf-8")


make_accuracy_outputs()
flow_run = find_flow_run()
print("Using flow run:", flow_run)

make_object_flow_collage(flow_run)
make_time_series_plots(flow_run)
update_readme()

print("\nDone. Updated:")
print(" -", README)
print(" -", ASSET_DIR / "accuracy_bar.png")
print(" -", ASSET_DIR / "object_flow_examples.png")
print(" -", ASSET_DIR / "time_series_example.png")
print(" -", ASSET_DIR / "class_average_time_series.png")
