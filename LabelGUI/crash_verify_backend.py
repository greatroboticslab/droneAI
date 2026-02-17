import os
import cv2
import time
import json
from datetime import timedelta
from typing import Optional, Dict, Any, List

import video_utils

# DB storage
try:
    from db.db_store import (
        init_db,
        upsert_video,
        get_latest_prediction,
        get_latest_verification,
        get_active_draft,
        create_verification,
        update_verification_counts,
        submit_verification,
        override_verification,
        list_verification_history,
    )
except Exception:
    # If DB folder not available, crash verification won't work.
    init_db = None

# -----------------------------------------------------------------------------
# Single active session at a time (keeps UI simple and stable)
# -----------------------------------------------------------------------------
_STATE: Dict[str, Any] = {
    "sid": None,

    # identity
    "video_id": None,
    "verification_id": None,
    "video_path": None,
    "person": None,
    "scenario": None,
    "youtube_link": None,

    # model prediction
    "pred_crash_events": 0,
    "pred_crashes_per_min": 0.0,

    # manual verification (live)
    "verified_crash_events": 0,
    "verified_times_sec": [],

    # playback
    "duration_sec": 0.0,
    "done": False,
    "saved": False,
    "last_error": None,

    # who is verifying
    "verifier": "",
    "role": "labeler",  # labeler | reviewer
}

def _repo_root() -> str:
    # LabelGUI/crash_verify_backend.py -> repo root is one level up
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _analysis_results_dir() -> str:
    return os.path.join(_repo_root(), "analysis", "results")

def _safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)

def _safe_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return int(default)

def get_state() -> Dict[str, Any]:
    return dict(_STATE)

def start_crash_verify_session(
    sid: str,
    video_path: str,
    person: str,
    scenario: str,
    youtube_link: str,
    verifier: str = "",
    role: str = "labeler",
    predicted_crash_events: Optional[int] = None,
    predicted_crashes_per_min: Optional[float] = None,
):
    """
    Starts (or resumes) a manual verification session.

    - Uses SQLite to store progress.
    - If the same verifier already has a DRAFT for this video, we resume it.
    - If role=reviewer and there is a latest submitted verification, we keep it as history
      and create a new draft that can override it upon submit.
    """
    global _STATE

    if init_db is None:
        _STATE["last_error"] = "DB import failed. Ensure db/db_store.py exists."
        return

    init_db()

    # Ensure the video exists in DB
    vid = upsert_video(person=person, scenario=scenario, youtube_link=youtube_link)

    # Pull latest model prediction if caller didn't provide
    if predicted_crash_events is None or predicted_crashes_per_min is None:
        pred = get_latest_prediction(vid) or {}
        if predicted_crash_events is None:
            predicted_crash_events = pred.get("pred_crash_events", 0)
        if predicted_crashes_per_min is None:
            predicted_crashes_per_min = pred.get("pred_crashes_per_min", 0.0)

    verifier = (verifier or "").strip()
    role = (role or "labeler").strip().lower()
    if role not in ("labeler", "reviewer"):
        role = "labeler"

    # Resume draft for this verifier if present
    draft = get_active_draft(vid, verifier) if verifier else None

    # If reviewer and there's a latest non-overridden verification, we want a fresh draft
    # (so submitting will override the previous "final" record).
    latest = get_latest_verification(vid)

    if role == "reviewer":
        draft = None  # reviewers always start a fresh draft

    if draft:
        verification_id = int(draft["id"])
        try:
            times = json.loads(draft.get("verified_times_json", "[]") or "[]")
        except Exception:
            times = []
        verified_events = _safe_int(draft.get("verified_crash_events", 0), 0)
    else:
        verification_id = create_verification(video_id=vid, verifier=verifier, role=role)
        times = []
        verified_events = 0

    # Setup in-memory state
    _STATE = {
        "sid": sid,
        "video_id": vid,
        "verification_id": verification_id,
        "video_path": video_path,
        "person": person,
        "scenario": scenario,
        "youtube_link": youtube_link,
        "pred_crash_events": _safe_int(predicted_crash_events, 0),
        "pred_crashes_per_min": _safe_float(predicted_crashes_per_min, 0.0),
        "verified_crash_events": int(verified_events),
        "verified_times_sec": list(times or []),
        "duration_sec": 0.0,
        "done": False,
        "saved": False,
        "last_error": None,
        "verifier": verifier,
        "role": role,
    }

def _persist_draft():
    """Persist current manual progress to SQLite (draft)."""
    if init_db is None:
        return
    if not _STATE.get("verification_id"):
        return
    update_verification_counts(
        verification_id=int(_STATE["verification_id"]),
        verified_crash_events=int(_STATE.get("verified_crash_events", 0)),
        verified_times=list(_STATE.get("verified_times_sec", [])),
        duration_sec=_STATE.get("duration_sec") or None,
    )

def mark_crash_now():
    """
    Count 1 crash at the current playback time.
    Also saves to DB draft immediately (so quitting the page doesn't lose progress).
    """
    now_sec = video_utils.get_current_time_sec()
    _STATE["verified_crash_events"] = int(_STATE.get("verified_crash_events", 0)) + 1
    _STATE.setdefault("verified_times_sec", []).append(float(now_sec))
    _persist_draft()

def undo_last_crash():
    times = _STATE.get("verified_times_sec") or []
    if times:
        times.pop()
        _STATE["verified_times_sec"] = times
        _STATE["verified_crash_events"] = max(0, int(_STATE.get("verified_crash_events", 0)) - 1)
        _persist_draft()

def save_and_label_later():
    """
    Save draft without submitting (user can come back later and resume).
    """
    _persist_draft()
    _STATE["saved"] = True

def finish_and_submit():
    """
    Submit verification. If reviewer, overrides the latest submitted verification.
    """
    if init_db is None:
        return

    # If reviewer, override current latest verification (history remains)
    if _STATE.get("role") == "reviewer":
        latest = get_latest_verification(int(_STATE["video_id"]))
        if latest and int(latest["id"]) != int(_STATE["verification_id"]):
            override_verification(int(latest["id"]))

    _persist_draft()
    submit_verification(int(_STATE["verification_id"]))
    _STATE["done"] = True
    _STATE["saved"] = True

def get_history() -> List[Dict[str, Any]]:
    if init_db is None or not _STATE.get("video_id"):
        return []
    return list_verification_history(int(_STATE["video_id"]))

# -----------------------------------------------------------------------------
# Video streaming (OpenCV -> MJPEG)
# -----------------------------------------------------------------------------
def generate_crash_video_stream():
    """
    Stream the selected video_path as MJPEG with a time overlay.
    """
    video_path = _STATE.get("video_path")
    if not video_path or not os.path.exists(video_path):
        _STATE["last_error"] = f"Video not found: {video_path}"
        yield b""
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _STATE["last_error"] = f"OpenCV failed to open: {video_path}"
        yield b""
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames and total_frames > 0:
        _STATE["duration_sec"] = float(total_frames / fps)
    else:
        _STATE["duration_sec"] = 0.0

    video_utils.set_video_duration(_STATE["duration_sec"])
    video_utils.set_current_time_sec(0.0)

    def overlay(frame, tsec):
        elapsed_str = str(timedelta(seconds=int(tsec)))
        total_str = str(timedelta(seconds=int(_STATE["duration_sec"] or 0)))
        cv2.putText(frame, f"{elapsed_str} / {total_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        # small hint for verifier
        cv2.putText(frame, f"Verified crashes: {int(_STATE.get('verified_crash_events', 0))}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, overlay):
        if not mjpeg_frame:
            time.sleep(0.05)
            continue
        yield mjpeg_frame

    # when video ends, keep a final draft save (so user doesn't lose last few clicks)
    _persist_draft()
