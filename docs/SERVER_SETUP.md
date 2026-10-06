# The labeling server

The team's NVIDIA PC runs the app for everyone. There is one database and one set of clips, on
that PC. Everyone labels from their own laptop in a web browser; nothing has to be merged.

| Who | Does what | Needs |
|---|---|---|
| The NVIDIA PC | Runs the app, keeps the DB, clips and videos, makes data snapshots, trains | The repo, `venv` (app) and `venv-ml` (training) |
| Labelers | Open the app in a browser and label | A browser, and the network link below |
| Anyone training elsewhere (e.g. on a Mac) | Downloads a data snapshot and trains on it | The repo and `venv-ml`; see [OPTICAL_FLOW_GUIDE.md](OPTICAL_FLOW_GUIDE.md) |

Several people can label at the same time: each person gets their own video, marks, pause and
playback position, and the queue never gives the same video to two people. Tested with two
people at once; each video stream uses about 0.3 MB/s.

## 1. Move the data to the PC (once)

Until now the labels have been made on one Mac. Stop the app there, then copy these folders to the
same place in the repo on the PC (a USB drive is the simplest):

| Copy | Size (Oct 2026) | Why |
|---|---|---|
| `db/droneai.sqlite` | under 1 MB | All labels, the queue, and the uploaded Excel datasets |
| `LabelGUI/ValidationResults/` | about 1.3 GB | The event clips cut while labeling |
| `LabelGUI/YouTubeDownloads/` | about 8.5 GB | The full videos. Needed to re-cut clips later, and YouTube videos do disappear |
| `LabelGUI/Uploads/videos/` | if it exists | Uploaded video files: the only copy |
| `analysis/data/test_videos.csv` | tiny | The held-out test videos. Must never change |
| `LabelGUI/TrainingRuns/`, `LabelGUI/OpticalFlowResults/` | a few MB | Earlier results, so they stay on the dashboard |

Paths inside the DB are relative to the repo, so the data works on Windows as is.

**From then on, label only on the PC's app.** Labels made in the app running on any other computer
stay on that computer and never reach the server.

## 2. Set up the PC

Follow **Setup → Windows** in the [README](../README.md) for the app (`venv`), and
[OPTICAL_FLOW_GUIDE.md](OPTICAL_FLOW_GUIDE.md) step 2 for training (`venv-ml`). Then:

```powershell
cd droneAI
venv\Scripts\Activate.ps1
python -m unittest discover -s LabelGUI/tests     # should end with OK
python LabelGUI/app.py
```

The app listens on port 5000 on every network card of the PC. The first time, Windows asks whether
Python may accept connections: allow it on **private** networks.

Keep the PC on while people label: **Settings → System → Power → Screen and sleep → When plugged
in, put my device to sleep after: Never**. If the app or the PC restarts, anyone mid-video sees
"This video is no longer open" and clicks **Resume** in the queue to start that video again.

## 3. Let people connect

**Same network (e.g. the lab Wi-Fi).** On the PC run `ipconfig` and read the IPv4 address, for
example `192.168.1.40`. Labelers open `http://192.168.1.40:5000`.

**From anywhere: Tailscale (recommended).** Tailscale is a free private network for small teams;
the app is never exposed to the internet.

1. Install Tailscale on the PC (https://tailscale.com/download) and sign in. Name the PC, e.g.
   `droneai-gpu` (Tailscale admin page → Machines → Edit name).
2. Invite each teammate (admin page → Users → Invite). They install Tailscale on their laptop and
   sign in.
3. Labelers open `http://droneai-gpu:5000`.

Do not forward port 5000 on the router to make the app public: the login is a shared team password
and the app was not built to face the internet.

## 4. Everyday use

- Log in with **your own name, spelled the same every time**. It is saved with every mark you make
  (reviewer), and the queue uses it to keep videos you have open locked to you.
- Click **Label next video →**. The queue hands each person a different video.
- The **Optical Flow Analysis** page shows how the model is doing and how many events are labeled.
- Avoid **DB Tools → Upload** on the server: it *replaces* the whole database with the uploaded file
  (a backup is saved first, but everyone's newer labels disappear from the app).

The Training and Crash Verify pages are still one-person-at-a-time: only one person should use
each of them at once.

## 5. Data snapshots for training

Labels change every day, so training uses a frozen copy. On the PC, before each training round:

```powershell
python LabelGUI/data_snapshot.py create --note "after the 100th video"
```

This writes `snapshots/v1_<date>/` and `snapshots/v1_<date>.zip` (the labeled clips, a copy of the
DB, the session-to-video map and the test split) and prints its **fingerprint**: a hash of every
label, clip and the test split. The next one is `v2`, and so on.

- Training on the PC: `python LabelGUI/extract_labeled_frames.py --snapshot snapshots/v1_<date>`, then
  the optical-flow and training commands from the guide. The run records `v1` and its fingerprint,
  and the dashboard shows them.
- Training elsewhere: download the zip from the **Data snapshots** table on the Optical Flow Analysis
  page, unzip it into `snapshots/` in your repo, and run the same commands.
- Check a downloaded snapshot is intact: `python LabelGUI/data_snapshot.py verify snapshots/v1_<date>`.

Two runs are directly comparable only if they show the same snapshot fingerprint. Never edit files
inside a snapshot; make a new one instead.

## 6. Backups

The PC now holds the only copy of the team's labels. Once a week, and before any big change:

1. **DB Tools → Download** the database, and keep it somewhere else (a cloud drive).
2. Copy `LabelGUI/ValidationResults/` and `LabelGUI/Uploads/` to an external drive.

The snapshot zips also contain the labeled clips and a DB copy, so keeping the newest one off the
PC covers most of it.
