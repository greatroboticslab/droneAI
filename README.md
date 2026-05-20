# DroneAI

**DroneAI** is a local web-based GUI for labeling drone flight videos. It helps users upload a video dataset, label flight events, track progress, create training data, and review crash results through a browser-based interface.

This version is focused on the **local application workflow**.

---

## Requirements

Before running DroneAI, make sure you have:

- **Python 3.10 or newer**
- **Git**
- **pip**
- **FFmpeg installed and added to PATH**
- **A virtual environment**
- **An Excel dataset file**

Your Excel file should include these columns:

```text
Persona Name
Youtube Link
```

---

## Setup

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/DroneAI.git
cd DroneAI
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment.

On Windows:

```bash
venv\Scripts\activate
```

On Mac/Linux:

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install flask pandas openpyxl opencv-python numpy yt-dlp paho-mqtt
```

If the project includes a `requirements.txt` file, you can also run:

```bash
pip install -r requirements.txt
```

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
- **Team Member** — can label videos from the queue

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

---

## Author

Developed by **Haider Baig**.
