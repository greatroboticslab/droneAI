# 🛰️ DroneAI Event Labeling & Crash Detection System

This repository provides a **Flask-based web GUI** to support labeling drone flight videos for research.  
It allows multiple event types to be labeled during video playback, saves short video clips around each event, and **organizes data automatically by participant**.  

✨ **New features** include Excel-based participant import, multiple event categories (take-off, landing, minor crash, severe crash), 5–10 second clips, and JSON-based labeling progress tracking with green/red status indicators.

---

## 🧭 Table of Contents

- [Features](#-features)
- [System Requirements](#-system-requirements)
- [Repository Structure](#-repository-structure)
- [1. Installation & Setup](#1-installation--setup)
  - [1.1 Clone the Repository](#11-clone-the-repository)
  - [1.2 Create a Virtual Environment](#12-create-a-virtual-environment)
  - [1.3 Install Dependencies](#13-install-dependencies)
- [2. Running the Application](#2-running-the-application)
  - [2.1 Starting the Server](#21-starting-the-server)
  - [2.2 Accessing the GUI](#22-accessing-the-gui)
- [3. Labeling Workflow](#3-labeling-workflow)
  - [3.1 Manual Labeling](#31-manual-labeling)
  - [3.2 Excel-Based Labeling](#32-excelbased-labeling)
  - [3.3 Event Categories](#33-event-categories)
  - [3.4 Output Structure](#34-output-structure)
- [4. Progress Tracking](#4-progress-tracking)
- [5. Folder & Clip Outputs](#5-folder--clip-outputs)
- [6. Developer Notes](#6-developer-notes)
  - [6.1 Code Overview](#61-code-overview)
  - [6.2 How Event Marking Works](#62-how-event-marking-works)
- [7. Troubleshooting](#7-troubleshooting)
- [8. License & Acknowledgments](#8-license--acknowledgments)

---

## 🚀 Features

- ✅ **Web-based GUI** with live video playback
- 🆕 **Multiple event categories**: Take-off, Landing, Minor Crash, Severe Crash
- 📝 Automatic **log file + short clips (10 frames)** per event
- 📂 **Folder structure organized by participant** (privacy-safe prefix naming)
- 📊 **Excel import** to preload labeling list
- 🟢🔴 **Green/red status tracking** for who’s labeled vs not
- 🧾 `progress.json` file for automatic tracking of labeling progress
- 🧠 Simple architecture — runs locally, no Docker required

---

## 🧰 System Requirements

- Python 3.10 or newer
- pip (Python package manager)
- Modern browser (Chrome, Edge, Firefox, Safari)
- (Optional) GPU for training downstream models (e.g., YOLOv8)

Required Python libraries:
```bash
Flask
pandas
openpyxl
yt-dlp
opencv-python
```

---

## 📁 Repository Structure

```
LabelGUI/
│
├── app.py                        # Main Flask application
├── validation_backend.py         # Video streaming + event logging
├── video_utils.py                # Pause/skip + frame management
├── training_backend.py           # (optional) training logic
│
├── templates/                    # HTML templates for GUI
│   ├── home.html
│   ├── validation_index.html
│   ├── validation_import.html    # Excel uploader
│   ├── validation_pick.html      # Participant picker + progress table
│   └── validation_results.html
│
├── static/                       # CSS and assets
│   └── style.css
│
├── YouTubeDownloads/             # Temp folder for downloaded videos
├── ValidationResults/            # All labeled outputs saved here
│   ├── progress.json             # Labeling progress tracking file
│   ├── SHON/                     # Example person folder
│   │   ├── Simulation 1/
│   │   │   ├── event_log.txt
│   │   │   ├── takeoff_01.mp4
│   │   │   └── landing_01.mp4
│   │   └── Real flight 1/
│   │       └── ...
│   └── ...
└── README.md
```

---

## 🛠 1. Installation & Setup

### 1.1 Clone the Repository
```bash
git clone https://github.com/<your_username>/droneAI.git
cd droneAI/LabelGUI
```

If you edited the repo on GitHub, make sure to pull the latest:
```bash
git fetch origin
git pull origin main
```

---

### 1.2 Create a Virtual Environment
```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash
# OR
source venv/bin/activate          # Mac/Linux
```

---

### 1.3 Install Dependencies
```bash
pip install -r requirements.txt
# or manually:
pip install flask pandas openpyxl yt-dlp opencv-python
```

---

## 🧭 2. Running the Application

### 2.1 Starting the Server
```bash
source venv/Scripts/activate   # (activate venv)
python app.py
```

You should see:
```
 * Running on http://127.0.0.1:5000
```

### 2.2 Accessing the GUI
Open your browser to:

```
http://localhost:5000/
```

---

## 🪂 3. Labeling Workflow

### 3.1 Manual Labeling
- From the home page, click **“Label Validation Data”**.  
- Enter:
  - YouTube video link  
  - Folder name (person’s name or ID)
- (Optional) check “Delete original downloaded video after processing”
- Click **Start Processing**  
- Watch the video and click event buttons when you see relevant moments:
  - Take-off
  - Landing
  - Minor Crash
  - Severe Crash
- Each click logs a timestamp and generates a short video clip (~10 frames before and after the event).

---

### 3.2 Excel-Based Labeling
- Click **“Import Excel & Pick Entry”** on the Validation page.
- Upload an Excel file containing at least:
  ```
  Persona Name | Youtube Link
  ```
- After upload, you’ll see a participant list with **green (labeled)** or **red (not yet labeled)** indicators.
- Select a person and scenario (Simulation or Real flight).
- Start labeling.

---

### 3.3 Event Categories

| Event Type       | Example Action                 | Clip Length         |
|------------------|----------------------------------|----------------------|
| Take-off         | Drone launches                  | ± 5 frames           |
| Landing          | Drone lands                     | ± 5 frames           |
| Minor Crash      | Soft collision / rough landing  | ± 5 frames           |
| Severe Crash     | Hard collision                  | ± 5 frames           |

---

### 3.4 Output Structure
For each session:
```
ValidationResults/
 └── SHON/
     └── Simulation 1/
         ├── event_log.txt
         ├── takeoff_01.mp4
         ├── landing_01.mp4
         └── ...
```

`event_log.txt` example:
```
YouTube Link: https://youtu.be/k8xM2VsClXg
Folder: ShontalBotros1

Take-off #1: [0:00:02 - 0:00:07]
Landing #2: [0:01:22 - 0:01:27]

Total Events Observed: 2
```

---

## 📊 4. Progress Tracking

The system automatically creates and maintains:

```
ValidationResults/progress.json
```

Example structure:
```json
{
  "updated_at": "2025-10-18T20:30:45.123Z",
  "people": {
    "SHON": {
      "full_names": ["Shontal Botros"],
      "sessions": [
        {
          "scenario": "Simulation",
          "folder": "ValidationResults/SHON/Simulation 1",
          "youtube_link": "https://youtu.be/example",
          "events": 2,
          "clips": 2,
          "timestamp": "2025-10-18T20:30:45.123Z"
        }
      ],
      "total_events": 2
    }
  }
}
```

- ✅ Progress is updated after each labeling session.
- 🟢 Green in GUI = already labeled at least once
- 🔴 Red in GUI = not yet labeled

---

## 🎥 5. Folder & Clip Outputs

For every event:
- A **short `.mp4` clip** is extracted
- The event name and index are overlaid on the clip
- A **single log file** is maintained per session

All outputs are stored under `ValidationResults/<first4letters>/<scenario N>/`.

> 💡 *Using first four letters of the name preserves privacy while keeping data organized.*

---

## 🧑‍💻 6. Developer Notes

### 6.1 Code Overview

| File                        | Purpose                                             |
|-----------------------------|-----------------------------------------------------|
| `app.py`                    | Flask routes, navigation, Excel import              |
| `validation_backend.py`     | Core video processing, event logging, progress save |
| `video_utils.py`            | Frame streaming, skip/pause                         |
| `templates/`                | Frontend HTML pages                                 |
| `progress.json`             | Labeling progress tracker                           |

---

### 6.2 How Event Marking Works
1. Flask serves MJPEG stream using `cv2.VideoCapture`.
2. User clicks event button → `/mark_event`.
3. Timestamp is recorded from `video_utils.get_current_time_sec()`.
4. ± N frames around the event are written to a new `.mp4` clip.
5. Event name + time range is appended to the log file.
6. Progress is updated in `progress.json`.

---

## 🛠 7. Troubleshooting

| Issue | Possible Cause | Fix |
|-------|----------------|-----|
| “Video not playing” | Wrong folder / missing YouTube download | Check console for errors |
| Flask 404 on `/import_excel` | Forgot to pull new code | Run `git pull origin main` |
| No clips saved | Event marking not triggered / permission error | Check write access in ValidationResults |
| `pandas` / `openpyxl` error | Missing dependencies | `pip install pandas openpyxl` |
| Old GUI showing | Browser caching | Hard refresh (Ctrl+F5) |

---

## 📜 8. License & Acknowledgments

- Built for **Drone Flight Research** at Middle Tennessee State University.  
- Developed with ❤️ using Python, Flask, and OpenCV.  
- Special thanks to research team contributors and labeling assistants.

---
