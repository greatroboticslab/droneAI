# Running optical flow and training

How to turn labeled clips into a trained event classifier on your own machine: a Mac with Apple
Silicon, or a Windows or Linux PC with an NVIDIA GPU. A computer without a GPU works too, about six
times slower.

The results show up on the **Optical-flow model** page of the app (`/optical_flow`).

## What the pipeline does

| Step | Command | Reads | Writes |
|---|---|---|---|
| 1. Frames | `extract_labeled_frames.py` | the DB and the event clips | `LabelGUI/FrameDataset/` |
| 2. Optical flow | `extract_dpflow_drone_features.py` | the frames | `LabelGUI/OpticalFlowResults/<run>/` |
| 3. Training | `train_dpflow_tabular_models.py` | the flow features | `LabelGUI/TrainingRuns/<run>/` |

1. **Frames.** Each labeled event has a short clip (2 s before the click, 1 s after). Step 1 keeps
   5 frames per second from every clip, so about 15 frames per event.
2. **Optical flow.** DPFlow measures how every pixel moves from one frame to the next. Those
   movements are summarised into numbers per frame pair: how fast the whole picture moves, which
   way, whether it zooms in (approaching the ground), spins, or freezes. If drone boxes from the
   detector exist (third-person videos), it also measures motion inside the box.
3. **Training.** Each clip's numbers are summarised (start, end, peaks, trends) into about 950
   values, and seven classifiers are compared with **5-fold cross-validation grouped by video**:
   all clips of a video are either in training or in testing, never both. Videos listed in
   `analysis/data/test_videos.csv` are never used.

Step 2 is the slow one. Steps 1 and 3 take a minute or two.

## 1. Get a data snapshot

Labeling happens on the team's server (see [SERVER_SETUP.md](SERVER_SETUP.md)), and the labels change
every day. Training always starts from a **data snapshot**: a frozen, numbered copy of the labeled
clips, the labels and the held-out test split. Everyone who trains on snapshot `v3` trains on exactly
the same data, and every run records the snapshot it used.

- **On the server:** make one with `python LabelGUI/data_snapshot.py create --note "why"`. It lands in
  `snapshots/`.
- **On your machine:** open the app's **Optical Flow Analysis** page, download the newest zip from the
  **Data snapshots** table, and unzip it into a `snapshots/` folder at the top of your repo. You should
  have `snapshots/v3_2026-10-12/snapshot.json`. Check it arrived intact:

  ```bash
  python LabelGUI/data_snapshot.py verify snapshots/v3_2026-10-12
  ```

A snapshot of 88 labeled videos is about 1.3 GB.

## 2. Install (once)

Install Python 3.11 or 3.12 and get the repo first; see **Setup** in the [README](../README.md).
Then make a separate environment for the ML packages (about 3 GB), named `venv-ml` so it stays
apart from the app's `venv`.

### Mac (Apple Silicon)

```bash
cd droneAI
python3.12 -m venv venv-ml
source venv-ml/bin/activate
pip install --upgrade pip
pip install -r LabelGUI/requirements-ml.txt
brew install libomp            # needed by xgboost on macOS
python -c "import torch; print('Apple GPU:', torch.backends.mps.is_available())"
```

It should print `Apple GPU: True`. You will use `--device mps`.

### Windows PC with an NVIDIA GPU (PowerShell)

```powershell
cd droneAI
py -3.12 -m venv venv-ml
venv-ml\Scripts\Activate.ps1
python -m pip install --upgrade pip
nvidia-smi                     # read "CUDA Version" in the top-right corner
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r LabelGUI/requirements-ml.txt
python -c "import torch; print('NVIDIA GPU:', torch.cuda.is_available())"
```

Pick `cu130`, `cu128` or `cu126` to match `nvidia-smi` (table in the README's **GPU setup**). The
check should print `NVIDIA GPU: True`. You will use `--device cuda`.

If PowerShell refuses to run `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

### Linux PC with an NVIDIA GPU

```bash
cd droneAI
python3.12 -m venv venv-ml
source venv-ml/bin/activate
pip install --upgrade pip
pip install -r LabelGUI/requirements-ml.txt
python -c "import torch; print('NVIDIA GPU:', torch.cuda.is_available())"
```

If it prints `False` and `nvidia-smi` shows a CUDA version below 13.0, install torch from the
`cu128` or `cu126` index first (README, **GPU setup**). You will use `--device cuda`.

## 3. Run it

Run every command from the `droneAI` folder with `venv-ml` activated. Replace
`v3_2026-10-12` with your snapshot, and pick a run name that says what changed, for example
`dpflow_fpv_v3`. Use the same name in steps 2 and 3 so the dashboard can link them.

**Mac / Linux**

```bash
python LabelGUI/extract_labeled_frames.py --snapshot snapshots/v3_2026-10-12

python LabelGUI/extract_dpflow_drone_features.py \
    --manifest LabelGUI/FrameDataset/frame_manifest.csv \
    --detections none \
    --device mps \
    --run-name dpflow_fpv_v3

python LabelGUI/train_dpflow_tabular_models.py \
    --features-csv LabelGUI/OpticalFlowResults/dpflow_fpv_v3/flow_sequence_features.csv \
    --view-type FPV \
    --run-name dpflow_fpv_v3
```

On Linux, use `--device cuda`.

**Windows (PowerShell)**: the line break character is a backtick.

```powershell
python LabelGUI/extract_labeled_frames.py --snapshot snapshots/v3_2026-10-12

python LabelGUI/extract_dpflow_drone_features.py `
    --manifest LabelGUI/FrameDataset/frame_manifest.csv `
    --detections none `
    --device cuda `
    --run-name dpflow_fpv_v3

python LabelGUI/train_dpflow_tabular_models.py `
    --features-csv LabelGUI/OpticalFlowResults/dpflow_fpv_v3/flow_sequence_features.csv `
    --view-type FPV `
    --run-name dpflow_fpv_v3
```

What each command does:

- `extract_labeled_frames.py --snapshot …` rebuilds `LabelGUI/FrameDataset/` from scratch using only
  the snapshot's DB and clips, and notes the snapshot id there. Videos labeled with no events have no
  clips and are listed as skipped; that is expected.
- The optical-flow step passes the snapshot id on, and training then uses that snapshot's
  session-to-video map and test split by itself. Each step prints `Data snapshot: v3 (…)`; if it says
  `none`, the frames did not come from a snapshot.
- `--detections none` runs without drone boxes, which is what FPV needs (the drone is the
  camera). For third-person videos with a detector output, pass that CSV instead.
- `--view-type FPV` trains on FPV clips only. Use `Third-person` or `all` for the others.

Before a long run, try a quick test with `--max-clips 4` added to the optical-flow command.

On the server you can also skip the snapshot and use the live data:
`python LabelGUI/video_manifest.py --active-only`, then `extract_labeled_frames.py --fresh`. Such runs
show **live data** on the dashboard and can't be compared with runs elsewhere.

### How long it takes

Step 2, for about 175 clips (about 2,300 frame pairs):

| Machine | Time |
|---|---|
| Mac, Apple GPU (`mps`) | about 15 s per clip, 40-45 min in total |
| Any computer, CPU only (`cpu`) | about 6 times slower than `mps` |
| NVIDIA GPU (`cuda`) | not measured yet; please add your numbers here |

The first run also downloads the DPFlow weights (38 MB) into `~/.cache/torch`, so it needs
internet once.

## 4. Read the results

Start the app (`python LabelGUI/app.py`) and open **Optical Flow Analysis** on the dashboard, or go
to http://localhost:5000/optical_flow. Pick your run at the top right. The page shows:

- accuracy with its spread across folds, next to what "always guess the most common event" would
  score and the 85% target
- per event: how many clips the model found, and how often it was right when it named that event
- what it mixes up, the score of each fold, how it was trained, the motion numbers it relied on,
  and the clips it got wrong with the most confidence (worth rewatching: sometimes the label is
  wrong)

The same numbers are in `LabelGUI/TrainingRuns/<run>/`: `metrics.json`, `classification_report.txt`,
`confusion_matrix.csv`, `predictions.csv`, `feature_importance.csv`.

To share a run, send the two small result folders, `LabelGUI/OpticalFlowResults/<run>/` (you can
leave out `debug_flow_images/`) and `LabelGUI/TrainingRuns/<run>/`. Copy them to the same place on
another machine and the run appears on its dashboard.

### Reading the score honestly

- Only compare runs marked **grouped by video**. Older runs used a clip or session split, where
  clips of one video were in both training and testing; the dashboard marks those "not
  comparable".
- Two runs are directly comparable only if they used the same data snapshot (same fingerprint in
  the **Data** column). A better score on a bigger snapshot may just be more data.
- With few videos, the score swings between folds (the ± value). A change smaller than the ± is
  probably noise.
- Accuracy alone can hide a lazy model: one that always says "Take-off" already scores about 40%.
  Check the macro F1 and the per-event table.

## Troubleshooting

- **`Could not read image` on every frame, then the run stops with exit code 138 (bus error).**
  The disk holding the repo dropped out for a moment. It happened once on an external USB SSD.
  Run again; if it keeps happening, move the repo to the internal disk.
- **`MPS requested, but no Apple GPU is available`.** An Intel Mac or an old macOS. Use
  `--device cpu`.
- **`CUDA requested, but torch.cuda.is_available() is False`.** You have the CPU build of torch.
  `pip uninstall -y torch torchvision` and install again from the CUDA index (step 2).
- **Training prints `only N videos ... using 3 folds instead of 5`.** Not enough labeled videos of
  some event for 5 folds. Label more, or accept fewer folds.
- **xgboost missing from the results on a Mac.** Run `brew install libomp` and train again.
- **The dashboard shows "No extraction summary found".** The training run points at a features
  file that isn't on this machine. Copy that `OpticalFlowResults/<run>/` folder over.
