# LabelGUI/crash_verify_backend.py
import os, csv, uuid, time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import video_utils
from validation_backend import download_video

from db.db_store import DBStore

BASE_DIR = Path(__file__).resolve().parent          # LabelGUI/
REPO_DIR = BASE_DIR.parent                          # DroneAI/
ANALYSIS_DIR = REPO_DIR / "analysis"
MANIFEST = ANALYSIS_DIR / "data" / "manifest.csv"
RESULTS_DIR = ANALYSIS_DIR / "results"
PER_VIDEO = RESULTS_DIR / "per_video.csv"
DOWNLOADS = ANALYSIS_DIR / "downloads"
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"

DOWNLOADS.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(REPO_DIR / "db").mkdir(parents=True, exist_ok=True)

db = DBStore(str(DB_PATH))

_ACTIVE: Dict[str, Dict[str, Any]] = {}  # sid -> session dict


def _read_manifest_rows() -> List[Dict[str, str]]:
    if not MANIFEST.exists():
        return []
    raw = MANIFEST.read_text(encoding="utf-8-sig")
    first = raw.splitlines()[0] if raw.splitlines() else ""
    delim = "\t" if "\t" in first and "," not in first else ","

    with open(MANIFEST, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delim)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
        for need in ("person", "scenario", "youtube_link"):
            if need not in cols:
                raise ValueError(f"manifest missing required column: {need}")

        rows = []
        for r in reader:
            person = (r[cols["person"]] or "").strip()
            scenario = (r[cols["scenario"]] or "").strip()
            link = (r[cols["youtube_link"]] or "").strip()
            if person and scenario and link:
                rows.append({"person": person, "scenario": scenario, "youtube_link": link})
        return rows


def _read_per_video() -> List[Dict[str, str]]:
    if not PER_VIDEO.exists():
        return []
    with open(PER_VIDEO, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def list_items_for_table() -> List[Dict[str, Any]]:
    manifest_rows = _read_manifest_rows()
    pv = _read_per_video()

    def match_pred(person, scenario, link):
        # exact match first
        for r in pv:
            if (r.get("youtube_link") or "").strip() == link.strip():
                return r
        # fallback match person+scenario
        for r in pv:
            if (r.get("person") or "").strip() == person.strip() and (r.get("scenario") or "").strip().lower() == scenario.strip().lower():
                return r
        return {}

    out = []
    for i, m in enumerate(manifest_rows):
        pred = match_pred(m["person"], m["scenario"], m["youtube_link"])
        pred_events = int(float(pred.get("crash_events", 0) or 0))
        pred_per_min = float(pred.get("crashes_per_min", 0) or 0)

        latest = db.get_latest_by_key(m["person"], m["scenario"], m["youtube_link"])
        verified = bool(latest and latest.get("status") in ("saved", "final"))

        out.append({
            "row_id": i,
            "person": m["person"],
            "scenario": m["scenario"],
            "youtube_link": m["youtube_link"],
            "pred_crashes": pred_events,
            "pred_per_min": f"{pred_per_min:.3f}",
            "verified": verified,
            "ver_crashes": (latest.get("verified_crash_events") if latest else ""),
            "ver_per_min": (f"{float(latest.get('verified_crashes_per_min', 0)):.3f}" if latest else ""),
            "sample_fps": "2",
            "conf": "0.5",
        })
    return out


def start_session(person: str, scenario: str, youtube_link: str, sample_fps: float, conf: float) -> str:
    sid = uuid.uuid4().hex[:12]

    # predicted values (if per_video.csv exists)
    pv = _read_per_video()
    pred = {}
    for r in pv:
        if (r.get("youtube_link") or "").strip() == youtube_link.strip():
            pred = r
            break

    pred_events = int(float(pred.get("crash_events", 0) or 0))
    pred_per_min = float(pred.get("crashes_per_min", 0) or 0)
    duration_sec = float(pred.get("duration_sec", 0) or 0)

    video_path = download_video(youtube_link, str(DOWNLOADS))
    if not video_path or not os.path.exists(video_path):
        raise RuntimeError("Failed to download video (yt-dlp/ffmpeg issue).")

    sess = {
        "sid": sid,
        "person": person,
        "scenario": scenario,
        "youtube_link": youtube_link,
        "video_path": video_path,
        "pred_crash_events": pred_events,
        "pred_crashes_per_min": pred_per_min,
        "duration_sec": duration_sec,
        "verified_crash_events": 0,
        "verified_crashes_per_min": 0.0,
        "notes": "",
        "resume_offset_sec": 0.0,
    }
    _ACTIVE[sid] = sess

    db.upsert_session(
        sid=sid,
        person=person,
        scenario=scenario,
        youtube_link=youtube_link,
        video_path=video_path,
        pred_crash_events=pred_events,
        pred_crashes_per_min=pred_per_min,
        duration_sec=duration_sec,
        verified_crash_events=0,
        verified_crashes_per_min=0.0,
        status="running",
        notes="",
        resume_offset_sec=0.0,
        sample_fps=float(sample_fps),
        conf=float(conf),
    )

    return sid


def get_session(sid: str) -> Dict[str, Any]:
    if sid in _ACTIVE:
        return _ACTIVE[sid]
    rec = db.get_session(sid)
    if not rec:
        raise KeyError("Session not found")
    _ACTIVE[sid] = dict(rec)
    return _ACTIVE[sid]


def stream_video(sid: str):
    sess = get_session(sid)

    cap = cv2.VideoCapture(sess["video_path"])
    if not cap.isOpened():
        yield b""
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    dur = (total_frames / fps) if total_frames else 0.0
    video_utils.set_video_duration(dur)
    video_utils.set_current_time_sec(0.0)

    def overlay(frame, tsec):
        a = f'{sess["person"]} | {sess["scenario"]}'
        b = f'Pred={sess["pred_crash_events"]}  Verified={sess["verified_crash_events"]}'
        cv2.putText(frame, a, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, b, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        return frame

    for chunk in video_utils.read_video_frames(cap, fps, overlay):
        if not chunk:
            time.sleep(0.05)
            continue
        yield chunk


def mark_plus_one(sid: str):
    sess = get_session(sid)
    sess["verified_crash_events"] = int(sess.get("verified_crash_events", 0)) + 1

    dur = float(sess.get("duration_sec", 0) or 0)
    if dur <= 0:
        dur = max(video_utils.get_current_time_sec(), 1.0)

    per_min = sess["verified_crash_events"] / (dur / 60.0)
    sess["verified_crashes_per_min"] = float(per_min)

    db.update_counts(sid, int(sess["verified_crash_events"]), float(sess["verified_crashes_per_min"]))


def save_label_later(sid: str, notes: str):
    resume_at = float(video_utils.get_current_time_sec())
    sess = get_session(sid)
    sess["notes"] = notes or ""
    sess["resume_offset_sec"] = resume_at
    db.update_status_notes(sid, "saved", sess["notes"], resume_at)
    return {"ok": True, "message": f"Saved ✅ (resume at {resume_at:.1f}s)"}


def finish_and_save(sid: str, notes: str):
    sess = get_session(sid)
    sess["notes"] = notes or ""
    db.update_status_notes(sid, "final", sess["notes"], 0.0)
    return {"ok": True, "message": "Final saved ✅"}


def export_excel(out_path: str):
    import pandas as pd
    rows = db.list_all()
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(out_path, index=False)
    return out_path
