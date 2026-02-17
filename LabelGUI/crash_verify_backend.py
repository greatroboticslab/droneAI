# LabelGUI/crash_verify_backend.py
import os
import cv2
import time
import json
from datetime import datetime, timedelta

import video_utils
from db.db_store import (
    init_db,
    get_or_create_video,
    add_inference_result,
    latest_inference,
    create_run,
    load_run,
    load_video,
    set_run_status,
    finish_run,
    increment_run_count,
    list_events,
)

# Initialize DB once (safe to call multiple times)
init_db()

# Single active session in-memory (streaming needs fast access)
_STATE = {
    "sid": None,
    "run_id": None,
    "video_id": None,
    "video_path": None,

    "person": None,
    "scenario": None,
    "youtube_link": None,

    "pred_crash_events": "",
    "pred_crashes_per_min": "",
    "duration_sec": 0.0,

    "done": False,
    "saved": False,
    "last_error": None,
}

def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _analysis_results_dir():
    return os.path.join(_repo_root(), "analysis", "results")

def start_crash_verify_session(
    sid: str,
    video_path: str,
    person: str,
    scenario: str,
    youtube_link: str,
    predicted_crash_events="",
    predicted_crashes_per_min="",
    duration_sec="",
    model_weights="analysis/weights/best.pt",
    sample_fps=None,
    conf=None,
    run_type="label",
    labeler=""
):
    """
    Creates a DB video row (if missing), saves inference result (optional),
    and creates a verification run for manual clicks.
    """
    global _STATE

    # Ensure consistent per-session time tracking
    video_utils.set_current_time_sec(0.0)
    video_utils.set_video_duration(0.0)

    video_id = get_or_create_video(person, scenario, youtube_link)

    # Save inference record (so Excel export can compare predicted vs verified)
    try:
        add_inference_result(
            video_id=video_id,
            crash_events=predicted_crash_events,
            crashes_per_min=predicted_crashes_per_min,
            duration_sec=duration_sec,
            video_path=video_path,
            model_weights=model_weights,
            sample_fps=sample_fps,
            conf=conf
        )
    except Exception:
        # best effort; verification can still happen
        pass

    run_id = create_run(video_id=video_id, run_type=run_type, labeler=labeler)

    _STATE = {
        "sid": sid,
        "run_id": run_id,
        "video_id": video_id,
        "video_path": video_path,

        "person": person,
        "scenario": scenario,
        "youtube_link": youtube_link,

        "pred_crash_events": str(predicted_crash_events).strip(),
        "pred_crashes_per_min": str(predicted_crashes_per_min).strip(),
        "duration_sec": float(duration_sec) if str(duration_sec).strip() != "" else 0.0,

        "done": False,
        "saved": False,
        "last_error": None,
    }

def resume_crash_verify_session(sid: str, run_id: int, video_path: str = None):
    """
    Resume an existing paused/in_progress run.
    """
    global _STATE

    run = load_run(run_id)
    if not run:
        return False, "Run not found."

    video = load_video(int(run["video_id"]))
    if not video:
        return False, "Video not found."

    inf = latest_inference(int(run["video_id"])) or {}

    _STATE = {
        "sid": sid,
        "run_id": int(run_id),
        "video_id": int(run["video_id"]),
        "video_path": video_path or (inf.get("video_path") or ""),

        "person": video.get("person"),
        "scenario": video.get("scenario"),
        "youtube_link": video.get("youtube_link"),

        "pred_crash_events": "" if inf.get("crash_events") is None else str(inf.get("crash_events")),
        "pred_crashes_per_min": "" if inf.get("crashes_per_min") is None else str(inf.get("crashes_per_min")),
        "duration_sec": float(inf.get("duration_sec") or 0.0),

        "done": False,
        "saved": (run.get("status") == "submitted"),
        "last_error": None,
    }

    # restore playback to last_time_sec
    last_t = float(run.get("last_time_sec") or 0.0)
    video_utils.set_current_time_sec(last_t)
    return True, "Resumed."

def mark_crash_now(sid: str):
    if _STATE.get("sid") != sid:
        return
    run_id = _STATE.get("run_id")
    if not run_id:
        return
    t = float(video_utils.get_current_time_sec() or 0.0)
    increment_run_count(run_id, t)

def save_and_label_later(sid: str, notes=""):
    """
    Save progress (paused) so user can resume later.
    """
    if _STATE.get("sid") != sid:
        return False, "Invalid session."

    run_id = _STATE.get("run_id")
    if not run_id:
        return False, "Missing run."

    last_t = float(video_utils.get_current_time_sec() or 0.0)
    set_run_status(run_id, "paused", last_time_sec=last_t, notes=notes)
    return True, "Saved progress (paused)."

def get_crash_verify_status(sid: str):
    if _STATE.get("sid") != sid:
        return {"ok": False, "error": "Invalid session."}

    run_id = _STATE.get("run_id")
    run = load_run(run_id) if run_id else None
    if not run:
        return {"ok": False, "error": "Run not found."}

    duration = float(_STATE.get("duration_sec") or 0.0)
    verified = int(run.get("verified_crash_events") or 0)
    verified_per_min = (verified / (duration / 60.0)) if duration > 0 else 0.0

    return {
        "ok": True,
        "done": bool(_STATE.get("done")),
        "saved": (run.get("status") == "submitted"),
        "paused": (run.get("status") == "paused"),

        "run_id": int(run_id),
        "person": _STATE.get("person"),
        "scenario": _STATE.get("scenario"),
        "youtube_link": _STATE.get("youtube_link"),

        "pred_crash_events": _STATE.get("pred_crash_events"),
        "pred_crashes_per_min": _STATE.get("pred_crashes_per_min"),

        "verified_crash_events": verified,
        "verified_crashes_per_min": round(verified_per_min, 3),
        "duration_sec": round(duration, 2),
        "last_time_sec": float(run.get("last_time_sec") or 0.0),

        "last_error": _STATE.get("last_error"),
    }

def generate_crash_verify_stream(sid: str):
    if _STATE.get("sid") != sid:
        yield b""
        return

    video_path = _STATE.get("video_path")
    if not video_path or not os.path.exists(video_path):
        _STATE["last_error"] = f"Video not found: {video_path}"
        _STATE["done"] = True
        yield b""
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        _STATE["last_error"] = f"OpenCV could not open video: {video_path}"
        _STATE["done"] = True
        yield b""
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    duration = (total_frames / fps) if total_frames > 0 else 0.0
    _STATE["duration_sec"] = float(duration)

    video_utils.set_video_duration(duration)

    # If resuming, start from stored time
    run = load_run(_STATE["run_id"]) if _STATE.get("run_id") else None
    start_time = float(run.get("last_time_sec") or 0.0) if run else 0.0
    if start_time > 0 and fps > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * fps))
        video_utils.set_current_time_sec(start_time)
    else:
        video_utils.set_current_time_sec(0.0)

    def draw_overlay(frame, current_time_sec):
        elapsed_str = str(timedelta(seconds=int(current_time_sec)))
        total_str = str(timedelta(seconds=int(duration)))

        run_id = _STATE.get("run_id")
        verified = 0
        if run_id:
            r = load_run(run_id)
            verified = int(r.get("verified_crash_events") or 0) if r else 0

        pred = _STATE.get("pred_crash_events", "")

        cv2.putText(frame, f"{elapsed_str} / {total_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, f"Verified crashes: {verified}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 255), 2)
        if str(pred).strip() != "":
            cv2.putText(frame, f"Predicted crashes: {pred}", (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 200, 0), 2)
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, draw_overlay):
        if not mjpeg_frame:
            time.sleep(0.05)
            continue

        # continuously update last_time_sec so resume works even if browser closes
        run_id = _STATE.get("run_id")
        if run_id:
            try:
                set_run_status(run_id, status="in_progress", last_time_sec=float(video_utils.get_current_time_sec() or 0.0))
            except Exception:
                pass

        yield mjpeg_frame

    _STATE["done"] = True

def finish_crash_verify_session(sid: str, notes=""):
    """
    Submit final results (status=submitted). No duplicate Excel rows: export pulls from DB.
    """
    if _STATE.get("sid") != sid:
        return False, "Invalid session."
    run_id = _STATE.get("run_id")
    if not run_id:
        return False, "Missing run."

    try:
        finish_run(run_id, notes=notes)
    except Exception as e:
        return False, f"Failed to submit: {e}"

    _STATE["saved"] = True
    return True, "Submitted verification results."

def get_run_events(run_id: int):
    """
    For exporting/reporting: list of crash click timestamps.
    """
    try:
        return list_events(run_id)
    except Exception:
        return []
