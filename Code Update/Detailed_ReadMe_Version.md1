# DroneAI Local GUI Guide

This guide explains how to use the current local version of **DroneAI**.

The purpose of this version is to provide a clean local workflow for uploading drone video datasets, labeling video events, tracking progress, creating training data, and reviewing crash results.

This guide focuses only on the **local application workflow**.

---

## What DroneAI Does

DroneAI is a local labeling tool for drone flight videos.

It allows users to:

- Upload an Excel dataset
- View videos in a queue
- Label flight events
- Track labeling progress
- Create training data
- Review crash analysis results
- Export or import the local database

The main goal is to make drone video labeling more organized and easier to manage.

---

## Starting the App

From the project folder, run:

```bash
python LabelGUI/app.py
```

Then open this address in your browser:

```text
http://localhost:5000
```

---

## Login Page

When the app opens, log in with:

- a username
- the shared password
- a role

Default password:

```text
droneai2025
```

There are two roles.

### Leader

A **leader** can upload and manage datasets.

Use this role if you are preparing the Excel dataset.

### Team Member

A **team member** can label videos from the existing queue.

Use this role if the dataset has already been uploaded.

---

## Dashboard

After logging in, you will see the dashboard.

The dashboard shows:

- your username
- your role
- the current dataset
- total number of videos
- number of unlabeled videos
- number of videos in progress
- number of labeled videos

This page gives a quick overview of the current labeling progress.

If no dataset has been uploaded yet, the dashboard will show that no dataset is selected.

---

## Dataset Manager

The **Dataset** page is mainly for leaders.

Use this page to upload an Excel file containing the video list.

To upload a dataset:

1. Click **Datasets**.
2. Enter a dataset name.
3. Choose the Excel file.
4. Select **Set as active dataset** if you want to use it immediately.
5. Click **Upload Dataset**.

After the upload, the dataset will appear under **Existing Datasets**.

Only the active dataset is used for the queue.

---

## Excel File Format

The Excel file should contain the video information.

The safest column names are:

```text
Persona Name
Youtube Link
```

Example:

| Persona Name | Youtube Link |
|---|---|
| Person 1 Trial 1 | https://www.youtube.com/watch?v=example |
| Person 1 Trial 2 | https://www.youtube.com/watch?v=example |

The app reads the Excel file and creates queue rows from it.

---

## Queue Page

The queue page shows the videos from the active dataset.

Each row includes:

- row number
- person or video name
- video link
- labeling status
- labeled by
- locked by
- start button

For the local version, this queue is stored in your local database.

---

## Queue Statuses

The queue uses these statuses:

### Not Labeled

The video has not been completed yet.

### In Progress

The video has been started.

### Labeled

The video has been completed.

---

## Labeling a Video

To label a video:

1. Open **Shared Queue**.
2. Choose a row with a YouTube video link.
3. Select the scenario type:
   - **Simulation**
   - **Real flight**
4. Leave **Delete original** unchecked for testing.
5. Click **Start**.

The validation page will open.

The video may take a few seconds to appear while it loads.

---

## Validation Page

The validation page is where event labeling happens.

You will see:

- the video stream
- playback controls
- event buttons
- event count

Playback controls include:

- **Rewind 10s**
- **Pause / Resume**
- **Forward 10s**

Event buttons include:

- **Take-off**
- **Land**
- **Minor Crash**
- **Severe Crash**

When you click an event button, the app records that event at the current video time.

The event count increases each time an event is marked.

---

## Completing a Validation Session

When the video finishes, the page will show a session complete message.

After completion:

1. Click **Return to Queue**.
2. Find the row you labeled.
3. Confirm that the status changed to **Labeled**.
4. Confirm that the **Labeled By** column shows your username.

The dashboard counts should also update.

For example, if the dataset has 95 videos and you label one video:

```text
Total: 95
Not Labeled: 94
Labeled: 1
```

The progress is saved in the local database, so it will still be there after logging out and logging back in.

---

## Training Page

The **Training** page is used to create labeled training data from videos.

This part of the app is separate from validation labeling.

Use this page when you want to collect labeled frames or segments for machine learning.

General workflow:

1. Open **Training**.
2. Enter the video link.
3. Choose or create a label group.
4. Start the training session.
5. Label the video as it plays.
6. Finish or pause the session as needed.

Training results are saved inside the project output folders.

---

## Crash Analysis Page

The **Crash Analysis** page is used to review crash-related results.

This section is for comparing predicted crash information with human verification.

Use this page to inspect crash counts, verify results, or export crash verification information.

---

## DB Tools

The **DB Tools** page is used to manage the local SQLite database.

The database file stores the dataset and labeling progress.

The database is located at:

```text
db/droneai.sqlite
```

From DB Tools, you can:

- download the current database
- import a previously exported database
- export database tables to Excel

This is useful for backing up progress or moving the current local state to another machine.

---

## Recommended Local Test

Use this checklist to confirm the app is working locally.

```text
[ ] App starts successfully
[ ] User can log in
[ ] Leader can upload dataset
[ ] Dataset becomes active
[ ] Dashboard shows correct video count
[ ] Queue shows dataset rows
[ ] User can start a video
[ ] Validation page loads
[ ] Event buttons work
[ ] Session completes
[ ] Queue item becomes labeled
[ ] Dashboard counts update
[ ] Progress remains after logout/login
```

If all of these pass, the local workflow is working correctly.

---

## Common Issues

### The video does not appear immediately

Wait a few seconds. The video may still be loading or downloading.

### Excel upload fails

Check that the Excel file has the expected columns:

```text
Persona Name
Youtube Link
```

### The app cannot process a video

Make sure **FFmpeg** is installed and added to PATH.

Also make sure the video link is accessible.

### Progress does not appear on another computer

The current version stores progress in the local SQLite database.

To move progress to another computer, export the database from **DB Tools** and import it on the other machine.

---

## Current Version Summary

This version of DroneAI supports:

- local login
- leader and team member roles
- dataset upload
- persistent local queue
- validation event labeling
- training data collection
- crash review
- database export/import

The current focus is to provide a stable local labeling workflow before adding stronger synchronization and future agent-based features.
