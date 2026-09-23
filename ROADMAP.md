# DroneAI Roadmap: AI Labeling with Optical Flow

Goal: the AI proposes drone events (takeoff, land, minor-crash, severe-crash) in full videos,
and a human confirms them. Target: event recall >= 95%, class accuracy >= 90%, and >= 95% on
auto-accepted events.

Full reasoning: the audit doc (DroneAI Audit: AI Labeling with Optical Flow).

## How to use this file (rules for Claude)

1. Work top to bottom. Pick the first unchecked task whose phase gate is not blocked.
2. One task at a time. When it is done and verified, change `- [ ]` to `- [x]` and append
   `(done YYYY-MM-DD, <short commit hash or "uncommitted">)`.
3. "Verified" means you ran something that proves it works (test, script output, app page loads).
   Write the evidence in one line under the task if it is not obvious.
4. Tasks tagged **[GPU]** run on the team's GPU PC, not this Mac. Build and dry-run the code here
   (tiny sample, `--device cpu`, `--max-clips 2`), check the box for the build, and leave the
   matching **[GPU run]** task open for the human to run.
5. Do not start a phase until the previous phase's gate is checked, unless the task says
   "can start early".
6. If a task turns out wrong or unnecessary, don't delete it: strike it through (`~~task~~`)
   and add one line saying why.
7. Update the **Status** line below whenever you check something off.

**Status:** Phase 0 in progress (4 tasks done). Next task: write `docs/LABELING_GUIDE.md`.

## Decisions (from the team, 2026-09-23)

- 50-100 new videos, 10 s to a few minutes each; a mix of YouTube links and video files.
- More than 15 of the videos contain landings.
- FPV is the main target. Third-person videos are the cross-check.
- YOLO stays in the pipeline (drone box for third-person, frame-state classifier for FPV).
- Minor crash = it crashes and can fly again. Severe crash = it can't fly anymore or is visibly broken.
- One person reviews pre-labels for now; the design must allow several reviewers later.
- Hand-reviewing about 30% of events is acceptable if the rest are auto-labeled at >= 95%.
- Heavy compute (DPFlow, YOLO, VideoMAE) runs on the GPU PC. This Mac is for building code.

## Phase 0: Foundation

- [x] Upload video files on the Datasets page (next to the Excel upload), stored in `LabelGUI/Uploads/videos/` (done 2026-09-23, uncommitted)
  Verified: `python -m unittest discover -s LabelGUI/tests` (10 pass) + Flask test-client run on an isolated copy.
- [x] Store `video_id`, `view_type`, `source` per video; read them from optional Excel columns `Video ID`, `View Type`, `Source` (done 2026-09-23, uncommitted)
  Verified: `python -m unittest discover -s LabelGUI/tests` (10 pass) + Flask test-client run on an isolated copy.
- [x] Labeling queue can open an uploaded local video instead of downloading from YouTube (done 2026-09-23, uncommitted)
  Verified: `python -m unittest discover -s LabelGUI/tests` (10 pass) + Flask test-client run on an isolated copy.
- [x] Fix MQTT `queue_claimed` handler using undefined variables (done 2026-09-23, uncommitted)
- [ ] Write the labeling guide `docs/LABELING_GUIDE.md` (the 4 classes, minor vs severe definition, when to click, view types)
- [ ] Fix `LabelGUI/requirements.txt`: split into `requirements-app.txt` (GUI) and `requirements-ml.txt` (torch, ptlflow, scikit-learn, ultralytics, transformers, paho-mqtt), with pinned versions tested on Python 3.11/3.12
- [ ] Replace hard-coded absolute/Windows paths in ML scripts and run metadata with paths relative to the repo root
- [ ] Add MPS (Apple GPU) as a device option next to cpu/cuda in the DPFlow extraction script, for small local runs
- [ ] Export a `video_manifest.csv` (video_id, view_type, source, path/link, status) from the DB, for the ML scripts to read
- [ ] Held-out test split: pick 20% of videos by `video_id` (mostly FPV, includes landings and both crash types), saved in `analysis/data/test_videos.csv`; never trained on
- [ ] Grouped 5-fold cross-validation by `video_id` in `train_dpflow_tabular_models.py`; report mean +- std; remove the clip split from reports
- [ ] **[GPU run]** Re-run the current DPFlow tabular model with grouped CV, FPV clips only, to get an honest baseline
- [ ] **Gate 0:** baseline number reproduced on the GPU PC from a clean checkout

## Phase 1: Data

- [ ] Upload all 50-100 videos (Excel for links, file upload for files) with view type and source set
- [ ] Label all videos in the GUI using `docs/LABELING_GUIDE.md` (can start early, in parallel with Phase 0)
- [ ] Re-cut clip script: rebuild clips from `validation_events` timestamps with a configurable window (default -2 s / +10 s), without relabeling
- [ ] Background windows: generate "nothing happening" windows from fully labeled videos, at least 3 s away from any event
- [ ] Record reviewer name on every accepted/edited event (multi-reviewer ready)
- [ ] Self-consistency check: the labeler re-labels a 20-event sample a week later
- [ ] **Gate 1:** >= 50 events per class, and the self-consistency check matches >= 90%

## Phase 2: Model v2

- [ ] FPV features: whole-frame DPFlow ego-motion (mean flow, rotation/curl, divergence for climb/drop, freeze detection)
- [ ] YOLOv8 frame-state classifier for FPV (in air / on ground / tumbling / static); outputs used as features
- [ ] Third-person features: keep YOLO drone box + DPFlow inside the box
- [ ] Two-stage crash model: detect crash from the impact, then judge severity from the after-window (flight resumes = minor; static/tilted/video ends = severe)
- [ ] Feature reduction: from 712 to a small, explainable set; confirm CV score holds
- [ ] 5-class classifier (4 events + background) with grouped CV
- [ ] **[GPU run]** Extract features for all clips and train
- [ ] Check that third-person results agree with FPV results on the same kinds of events
- [ ] Fallback if stuck below 85%: fine-tune VideoMAE on FPV clips and combine with flow features
- [ ] **Gate 2:** grouped-CV class accuracy >= 85%

## Phase 3: Finding events in full videos

- [ ] Sliding-window scorer: 3 s windows, 0.5 s step, over a full video
- [ ] Peak picking: merge nearby windows into proposed events with a time and confidence
- [ ] Event-level evaluation: a proposal counts if the right class lands within +-1 s of the human mark; precision/recall per class
- [ ] **[GPU run]** Run on the held-out test videos
- [ ] **Gate 3:** >= 90% of events found within +-1 s on test videos

## Phase 4: Pre-labeling in the GUI

- [ ] Load proposed events as markers on the labeling page; jump between them
- [ ] Accept / fix / reject per marker; saved to `validation_events` with reviewer name
- [ ] Confidence gate: auto-accept above a threshold tuned for >= 95% accuracy; report coverage
- [ ] Retrain loop: retrain every 10-20 newly reviewed videos
- [ ] **Gate 4:** recall >= 95%, class accuracy >= 90%, >= 95% on auto-accepted events

## Known issues to fix along the way

- [ ] Session state lives in module globals in `validation_backend.py` / `training_backend.py`: two labelers on one server overwrite each other
- [ ] Secret key and shared password are hard-coded in `LabelGUI/app.py`; move to environment variables
- [ ] Model weights and `.pkl` files are committed to git; move to releases or external storage
- [ ] Duplicate scripts in `AI_Work/scripts/` and `LabelGUI/`; keep one copy
- [ ] Uploaded video files are not synced over MQTT to other machines (only Excel datasets are)
