# DroneAI

**DroneAI** is a local web-based GUI for labeling drone flight videos. It helps users upload a video dataset, label flight events, track progress, create training data, and review crash results through a browser-based interface.

This first version is focused on the **local application workflow**.

---

## Requirements

- **Python 3.11 or 3.12.** Python 3.10 is too old for the pinned packages. 3.13 and newer are not tested.
- **Git**
- **An Excel dataset file** for the video links (or video files to upload)
- **For the ML scripts only:** an NVIDIA GPU is strongly recommended. See [GPU setup](#gpu-setup-ml-scripts).

FFmpeg is optional. If it is not on your PATH, the app uses the copy bundled with the
`imageio-ffmpeg` package.

Your Excel file must include these columns:

```text
Persona Name
Youtube Link
```

Optional columns: `Video ID`, `View Type` (FPV / Third-person), `Source` (Real / Simulation).

---

## Setup

There are two install levels:

| Install | File | Use it for |
|---|---|---|
| App | `LabelGUI/requirements-app.txt` | The labeling GUI. Enough for labelers. |
| ML | `LabelGUI/requirements-ml.txt` | The GUI plus optical flow, training, YOLO. Includes the app file. |

Pick your operating system below. Every command runs from the repository folder.

### macOS

1. Install Homebrew from https://brew.sh if you don't have it. Then install Python and OpenMP:

   ```bash
   brew install python@3.12 libomp git
   ```

   `libomp` is needed by xgboost. Without it, the tabular training script skips the xgboost model.

2. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/greatroboticslab/droneAI.git
   cd droneAI
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install the packages:

   ```bash
   pip install --upgrade pip
   pip install -r LabelGUI/requirements-app.txt   # GUI only
   # or
   pip install -r LabelGUI/requirements-ml.txt    # GUI + ML
   ```

Macs have no NVIDIA GPU, but PyTorch can use the Apple GPU (MPS) on Apple Silicon Macs. The
DPFlow feature extraction script accepts `--device mps` for small local runs, for example:

```bash
python LabelGUI/extract_dpflow_drone_features.py --device mps --max-clips 2
```

On a small test it gave the same numbers as `--device cpu` and ran about 6 times faster. The
other ML scripts still accept only `cpu` or `cuda`. Run full-size jobs on the GPU PC.

### Linux (Ubuntu / Debian)

1. Install Python and the system libraries OpenCV needs:

   ```bash
   sudo apt update
   sudo apt install -y git python3.12 python3.12-venv libgl1 libglib2.0-0
   ```

   Ubuntu 24.04 ships Python 3.12. On older releases, install 3.11 or 3.12 from the
   deadsnakes PPA and use that version in the next step.

2. Clone the repository and create a virtual environment:

   ```bash
   git clone https://github.com/greatroboticslab/droneAI.git
   cd droneAI
   python3.12 -m venv venv
   source venv/bin/activate
   ```

3. Install the packages:

   ```bash
   pip install --upgrade pip
   pip install -r LabelGUI/requirements-app.txt   # GUI only
   # or
   pip install -r LabelGUI/requirements-ml.txt    # GUI + ML
   ```

   On Linux the default PyTorch from pip is already a CUDA build (CUDA 13.0). If your NVIDIA
   driver is older, see [GPU setup](#gpu-setup-ml-scripts) before installing the ML file.

### Windows

1. Install Python 3.12 from https://www.python.org/downloads/windows/. In the installer, tick
   **"Add python.exe to PATH"**. Install Git from https://git-scm.com/download/win.

2. Open **PowerShell**, clone the repository and create a virtual environment:

   ```powershell
   git clone https://github.com/greatroboticslab/droneAI.git
   cd droneAI
   py -3.12 -m venv venv
   venv\Scripts\Activate.ps1
   ```

   If PowerShell says running scripts is disabled, run this once and then activate again:

   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
   ```

   In the old Command Prompt (cmd), activate with `venv\Scripts\activate.bat` instead.

3. Install the packages. For the GUI only:

   ```powershell
   python -m pip install --upgrade pip
   pip install -r LabelGUI\requirements-app.txt
   ```

   For the ML scripts on a GPU PC, **install the CUDA build of PyTorch first**. The default
   PyTorch from pip on Windows is CPU-only and would make every GPU job run on the CPU. Follow
   [GPU setup](#gpu-setup-ml-scripts), then:

   ```powershell
   pip install -r LabelGUI\requirements-ml.txt
   ```

### GPU setup (ML scripts)

Heavy ML jobs (DPFlow optical flow, YOLO, VideoMAE) need an NVIDIA GPU with CUDA.

1. **Install or update the NVIDIA driver.** Get it from https://www.nvidia.com/Download/index.aspx
   (on Ubuntu, `sudo ubuntu-drivers autoinstall` also works). You don't need to install the CUDA
   Toolkit. The PyTorch packages bring their own CUDA libraries.

2. **Check which CUDA version your driver supports:**

   ```bash
   nvidia-smi
   ```

   Read `CUDA Version` in the top-right corner. It is the highest CUDA version the driver can run.

3. **Pick the matching PyTorch build:**

   | `CUDA Version` in nvidia-smi | Index to use |
   |---|---|
   | 13.0 or newer | `cu130` |
   | 12.8 or 12.9 | `cu128` |
   | 12.6 or 12.7 | `cu126` |
   | Older than 12.6 | Update the driver first |

4. **Install PyTorch from that index** inside the activated virtual environment, before the ML
   file. Replace `cu128` with your index:

   ```bash
   pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
   pip install -r LabelGUI/requirements-ml.txt
   ```

   On Windows this step is always needed. On Linux it is only needed when your driver is older
   than CUDA 13.0, because the default Linux build already uses CUDA 13.0.

5. **Check that PyTorch sees the GPU:**

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
   ```

   It should print `True` and your GPU name. If it prints `False`, check the troubleshooting list
   below.

6. Run the ML scripts with `--device cuda`.

### Check the install

Run the unit tests. They use temporary databases and don't touch your data:

```bash
python -m unittest discover -s LabelGUI/tests
```

### Troubleshooting

- **`torch.cuda.is_available()` is `False` on Windows.** You have the CPU build. Run
  `pip uninstall -y torch torchvision`, then repeat step 4 of GPU setup.
- **`torch.cuda.is_available()` is `False` on Linux.** Run `nvidia-smi`. If it fails, the driver
  is missing. If its CUDA version is below 13.0, reinstall PyTorch from the `cu128` or `cu126`
  index (GPU setup, step 4).
- **xgboost error about `libomp.dylib` on macOS.** Run `brew install libomp`.
- **`ImportError: libGL.so.1` on Linux.** Run `sudo apt install -y libgl1 libglib2.0-0`.
- **YouTube downloads start failing.** YouTube changes often. Update yt-dlp with
  `pip install -U yt-dlp`.

---

## Run the Application

Start the Flask app:

```bash
python LabelGUI/app.py
```

Then open this address in your browser:

```text
http://localhost:5000
```

---

## Login

Use any username.

Default password:

```text
droneai2025
```

Choose a role:

- **Leader** — can upload and manage datasets
- **Team Member** — can label videos from the queue (If you want to label videos, just log in as team member)

---

## Basic Local Workflow

1. Log in as a **leader**.
2. Open **Datasets**.
3. Upload the Excel dataset.
4. Set the dataset as active.
5. Open the queue.
6. Start a video row.
7. Mark events while the video plays.
8. Return to the queue after the session finishes.
9. Confirm the row is marked as labeled.

---

## Main Pages

- **Dashboard** — shows the current dataset and labeling progress
- **Datasets** — upload and manage Excel datasets
- **Shared Queue** — select videos to label
- **Training** — create labeled training data
- **Crash Analysis** — review crash verification results
- **DB Tools** — export or import the local database

---

## Local Database

Progress is saved in the local SQLite database:

```text
db/droneai.sqlite
```

If you want another machine to continue from the same state, export the database from **DB Tools** and import it on the other machine.

To give the ML scripts a list of videos, export a video manifest (one row per video, with its
view type, source, link or file path, labeling status and event counts):

```bash
python LabelGUI/video_manifest.py                  # writes analysis/data/video_manifest.csv
python LabelGUI/video_manifest.py --active-only    # only the active dataset
```

The database is opened read-only. Only finished labeling sessions are counted.

---

## Author

Developed by **Haider Baig**.
