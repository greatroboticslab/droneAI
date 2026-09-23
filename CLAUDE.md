# DroneAI

Local Flask labeling GUI for drone flight videos (`LabelGUI/app.py`) plus optical-flow ML scripts
that learn to label takeoff / land / minor-crash / severe-crash.

## Follow the roadmap

`ROADMAP.md` is the plan of record. Before starting work, read it, take the first unchecked task,
and follow the rules at its top: one task at a time, verify, then tick the box with the date and
commit hash and update the Status line.

## Working notes

- Run the app: `python LabelGUI/app.py` -> http://localhost:5000 (password in `app.py`).
- Tests: `python -m unittest discover -s LabelGUI/tests` (no pytest needed).
- The live DB is `db/droneai.sqlite` at the repo root. Never test against it; tests use temp DBs.
  Importing `app.py` opens that DB, so run app-level checks on a copy of `LabelGUI/` in a temp folder.
- Excel and file parsing lives in `LabelGUI/dataset_import.py` (testable without Flask/MQTT).
- Uploaded videos go to `LabelGUI/Uploads/videos/` (gitignored), named `file-<sha256 prefix>`.
- Heavy ML (DPFlow, YOLO, VideoMAE) runs on the team's GPU PC. On this Mac, build and dry-run only.
- MQTT publishes to a public broker (`broker.hivemq.com`); don't publish test data.
