import os
import re
import cv2
import yt_dlp
import threading
import time
import json
import uuid
from datetime import timedelta, datetime
from pathlib import Path

import pandas as pd  # pip install pandas openpyxl
import video_utils

from db.db_store import DBStore

# -----------------------
# Paths / DB
# -----------------------
BASE_DIR = Path(__file__).resolve().parent      # LabelGUI/
REPO_DIR = BASE_DIR.parent                     # DroneAI/
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"
db = DBStore(str(DB_PATH))

# -----------------------
# Globals / session state
# -----------------------
_processing_thread = None
_video_done = False
_log_file_path = None
_delete_original = False
_event_times = []          # list of tuples: (idx, event_type, time_sec)
_video_duration = 0.0
_current_video_file = None
_validation_sid = None
_target_folder = None
_last_youtube_link = ""

_extraction_in_progress = False
_extraction_current = 0
_extraction_total = 0


# -----------------------
# URL normalization
# -----------------------
def _normalize_youtube_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    m = re.match(r"^https?://youtu\.be/([A-Za-z0-9_-]{8,})", url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    m = re.match(r"^https?://(www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{8,})", url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(2)}"
    return url


# -----------------------
# Public API used by app.py
# -----------------------
def start_validation_thread(
    youtube_link,
    folder_name=None,
    delete_original=False,
    person_name=None,
    scenario_base=None  # "Simulation" or "Real flight"
):
    """
    If person_name & scenario_base provided:
        ValidationResults/<first 4 chars>/<scenario_base N>/
    Else:
        ValidationResults/<unique folder_name>/
    """
    global _processing_thread, _video_done, _log_file_path
    global _delete_original, _event_times, _video_duration, _current_video_file
    global _validation_sid, _target_folder, _last_youtube_link

    # Stop any prior session cleanly
    if _processing_thread and _processing_thread.is_alive():
        _video_done = True

    _video_done = False
    _event_times = []
    _video_duration = 0.0
    _delete_original = bool(delete_original)
    _last_youtube_link = (youtube_link or "").strip()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, "YouTubeDownloads")
    os.makedirs(youtube_downloads_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, "ValidationResults")
    os.makedirs(results_dir, exist_ok=True)

    # Build target folder
    if person_name and scenario_base:
        prefix = (person_name or "").strip()[:4] or "User"
        person_dir = os.path.join(results_dir, prefix)
        os.makedirs(person_dir, exist_ok=True)

        existing = [d for d in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, d))]
        count = sum(1 for d in existing if d.lower().startswith((scenario_base or "").lower()))
        scenario_dir_name = f"{scenario_base} {count + 1}"
        target_folder = os.path.join(person_dir, scenario_dir_name)
        os.makedirs(target_folder, exist_ok=True)
    else:
        folder_name = folder_name or "Session"
        target_folder = get_unique_folder_name(results_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)

    _target_folder = target_folder
    _log_file_path = os.path.join(target_folder, "event_log.txt")

    # Create DB session
    _validation_sid = str(uuid.uuid4())

    db.upsert_validation_session(
        sid=_validation_sid,
        person_name=(person_name or ""),
        scenario_base=(scenario_base or ""),
        youtube_link=_last_youtube_link,
        folder_path=os.path.relpath(target_folder, start=str(REPO_DIR)),
        delete_original=1 if _delete_original else 0,
        status="running",
        duration_sec=0.0,
        events_count=0,
    )

    # Download
    downloaded_filepath = download_video(youtube_link, youtube_downloads_dir)
    _current_video_file = downloaded_filepath

    if not downloaded_filepath:
        _video_done = True
        db.finalize_validation_session(_validation_sid, 0.0, 0, status="failed")
        with open(_log_file_path, "w", encoding="utf-8") as f:
            f.write("Download failed.\n")
        return

    # Update DB with video_path now that we have it
    db.upsert_validation_session(
        sid=_validation_sid,
        youtube_link=_last_youtube_link,
        video_path=os.path.relpath(downloaded_filepath, start=str(REPO_DIR)),
        status="running",
    )

    def video_thread():
        nonlocal downloaded_filepath, target_folder, youtube_link, base_dir

        with open(_log_file_path, "w", encoding="utf-8") as f:
            f.write(f"YouTube Link: {youtube_link}\n")
            f.write(f"Folder: {os.path.relpath(target_folder, base_dir)}\n\n")

        # Wait until user finishes watching/marking OR stream ends
        while not _video_done:
            time.sleep(0.5)

        # Extract clips after marking ends
        cap_for_fps = cv2.VideoCapture(downloaded_filepath)
        fps = (cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0) if cap_for_fps.isOpened() else 30.0
        cap_for_fps.release()

        multiple_pass_extract(downloaded_filepath, target_folder, _event_times, fps)

        # Optionally delete original
        if _delete_original and os.path.exists(downloaded_filepath):
            try:
                os.remove(downloaded_filepath)
            except Exception:
                pass

        # Log summary
        with open(_log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\nTotal Events Observed: {len(_event_times)}\n")

        # Update progress.json (Excel-driven sessions only)
        try:
            update_progress_record(
                person_name=person_name,
                youtube_link=youtube_link,
                scenario_base=scenario_base,
                target_folder=target_folder,
                events_count=len(_event_times),
            )
        except Exception:
            pass

        # Finalize DB session
        db.finalize_validation_session(
            _validation_sid,
            duration_sec=float(_video_duration or 0.0),
            events_count=int(len(_event_times)),
            status="final"
        )

        finalize_video(target_folder)

    _processing_thread = threading.Thread(target=video_thread, daemon=True)
    _processing_thread.start()


def generate_video_stream():
    """
    Streams the current downloaded video via MJPEG frames.
    Single loop only (no duplicates).
    """
    global _video_done, _video_duration, _current_video_file

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, "YouTubeDownloads")

    if _current_video_file and os.path.exists(_current_video_file):
        candidate = _current_video_file
    else:
        try:
            files = sorted(
                [os.path.join(youtube_downloads_dir, f) for f in os.listdir(youtube_downloads_dir)],
                key=os.path.getmtime,
            )
        except Exception:
            files = []
        candidate = None
        for f in reversed(files):
            if f.lower().endswith(".mp4"):
                candidate = f
                break

    if not candidate:
        while not _video_done:
            time.sleep(0.2)
            yield b""
        return

    cap = cv2.VideoCapture(candidate)
    if not cap.isOpened():
        while not _video_done:
            time.sleep(0.2)
            yield b""
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    _video_duration = (total_frames / fps) if total_frames else 0.0

    video_utils.set_video_duration(_video_duration)
    video_utils.set_current_time_sec(0.0)

    def draw_overlay(frame, current_time_sec):
        elapsed = video_utils.format_time(current_time_sec)
        total = video_utils.format_time(_video_duration)
        cv2.putText(
            frame,
            f"{elapsed} / {total}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
        )
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, draw_overlay):
        if not mjpeg_frame:
            time.sleep(0.05)
            continue
        yield mjpeg_frame

    _video_done = True


def mark_event_now(event_type: str):
    """
    Called by /mark_event endpoint.
    Logs into memory AND DB.
    """
    global _event_times, _validation_sid

    idx = len(_event_times) + 1
    current_time_sec = video_utils.get_current_time_sec()
    _event_times.append((idx, event_type, current_time_sec))

    if _validation_sid:
        db.insert_validation_event(_validation_sid, idx, event_type, current_time_sec)


def is_video_done():
    return _video_done


def finalize_video(target_folder):
    global _video_done
    _video_done = True


def get_crash_count():
    return len(_event_times)


def get_extraction_progress():
    return {
        "in_progress": _extraction_in_progress,
        "current": _extraction_current,
        "total": _extraction_total,
    }


def toggle_pause():
    return video_utils.toggle_pause_flag()


def skip_video(offset_seconds: float):
    video_utils.schedule_skip(offset_seconds)


def get_logged_events():
    results = []
    for (idx, event_type, ctime) in _event_times:
        results.append({
            "index": idx,
            "type": event_type,
            "start": sec_to_hms(max(0, ctime - 1)),
            "end": sec_to_hms(ctime + 1),
        })
    return results


# -----------------------
# Extraction (NO DUPLICATES)
# -----------------------
def multiple_pass_extract(video_path, target_folder, event_times_list, fps_hint):
    global _extraction_in_progress, _extraction_current, _extraction_total, _log_file_path

    if not event_times_list:
        return

    sorted_times = sorted(event_times_list, key=lambda x: x[2])

    _extraction_in_progress = True
    _extraction_current = 0
    _extraction_total = len(sorted_times)

    excel_rows = []

    with open(_log_file_path, "a", encoding="utf-8") as lf:
        for (idx, event_type, ctime) in sorted_times:
            _extraction_current += 1

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
            event_frame = int(ctime * fps)
            frame_window = 15

            start_frame = max(0, event_frame - frame_window)
            end_frame = event_frame + frame_window

            lf.write(
                f"{event_type.capitalize()} #{idx}: [Frames {start_frame}-{end_frame}] "
                f"(~{sec_to_hms(start_frame/fps)} - ~{sec_to_hms(end_frame/fps)})\n"
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            out_filename = os.path.join(target_folder, f"{event_type}_{idx:02d}.mp4")
            writer = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                if current_frame_idx > end_frame:
                    break

                if writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(out_filename, fourcc, fps, (w, h))

                overlay_text = f"{event_type.capitalize()} #{idx}, Frame={current_frame_idx}"
                frame_copy = frame.copy()
                cv2.putText(frame_copy, overlay_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                writer.write(frame_copy)

            if writer:
                writer.release()
            cap.release()

            excel_rows.append({
                "Event index": idx,
                "Event type": event_type,
                "Event time (sec)": round(ctime, 3),
                "Event time (hh:mm:ss)": sec_to_hms(ctime),
                "Start frame": start_frame,
                "End frame": end_frame,
                "Approx start time": sec_to_hms(start_frame / fps),
                "Approx end time": sec_to_hms(end_frame / fps),
                "Clip filename": os.path.basename(out_filename),
            })

    _extraction_in_progress = False

    if excel_rows:
        base_title = os.path.splitext(os.path.basename(video_path))[0]
        excel_name = f"{base_title}_labels.xlsx"
        excel_path = os.path.join(target_folder, excel_name)
        try:
            df = pd.DataFrame(excel_rows)
            df.to_excel(excel_path, index=False)
        except Exception as e:
            print("[multiple_pass_extract] Failed to write Excel file:", e)


# -----------------------
# Helpers
# -----------------------
def sec_to_hms(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def get_unique_folder_name(parent_dir, base_name):
    candidate = os.path.join(parent_dir, base_name)
    if not os.path.exists(candidate):
        return candidate
    counter = 1
    while True:
        new_candidate = os.path.join(parent_dir, f"{base_name}{counter}")
        if not os.path.exists(new_candidate):
            return new_candidate
        counter += 1


# -----------------------
# Download (clean)
# -----------------------
def download_video(youtube_link, download_folder):
    os.makedirs(download_folder, exist_ok=True)
    youtube_link = _normalize_youtube_url(youtube_link)

    # NOTE: keep your existing path (we can move this later)
    FFMPEG_DIR = r"C:\Users\rusha\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

    base_opts = {
        "outtmpl": os.path.join(download_folder, "%(title).50s-%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_DIR,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "retries": 5,
        "concurrent_fragment_downloads": 4,
    }

    strategies = [
        "bv*+ba/bestvideo*+bestaudio",
        "best[ext=mp4]",
        "18",
        "best",
    ]

    def _resolve_output_paths(ydl, info):
        paths = []
        for k in ("requested_downloads", "requested_formats", "files"):
            lst = info.get(k) or []
            for item in lst:
                fp = item.get("filepath") or item.get("_filename")
                if fp and os.path.exists(fp):
                    paths.append(fp)

        fn = info.get("_filename")
        if fn and os.path.exists(fn):
            paths.append(fn)

        try:
            prepared = ydl.prepare_filename(info)
            if prepared and os.path.exists(prepared):
                paths.append(prepared)
            base, _ = os.path.splitext(prepared)
            mp4 = base + ".mp4"
            if os.path.exists(mp4):
                paths.append(mp4)
        except Exception:
            pass

        seen = set()
        uniq = []
        for p in paths:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq

    def try_with(fmt):
        opts = dict(base_opts)
        opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_link, download=True)
                out_paths = _resolve_output_paths(ydl, info)
                for p in out_paths:
                    if p.lower().endswith((".mp4", ".mkv", ".webm")) and os.path.exists(p):
                        return p
        except Exception as e:
            print(f"[download_video] attempt with '{fmt}' failed:", e)
        return None

    for fmt in strategies:
        p = try_with(fmt)
        if p:
            return p

    return None


# -----------------------
# Progress tracking (Excel-driven sessions)
# -----------------------
def _progress_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "ValidationResults")
    os.makedirs(results_dir, exist_ok=True)
    return os.path.join(results_dir, "progress.json")


def _load_progress():
    path = _progress_path()
    if not os.path.exists(path):
        return {"updated_at": None, "people": {}}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"updated_at": None, "people": {}}


def _save_progress(data):
    data["updated_at"] = datetime.utcnow().isoformat()
    with open(_progress_path(), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _count_clips(folder):
    try:
        return sum(1 for f in os.listdir(folder) if f.lower().endswith(".mp4"))
    except Exception:
        return 0


def update_progress_record(person_name, youtube_link, scenario_base, target_folder, events_count):
    if not person_name or not scenario_base:
        return

    prefix = (person_name or "").strip()[:4] or "User"
    data = _load_progress()

    if "people" not in data:
        data["people"] = {}
    if prefix not in data["people"]:
        data["people"][prefix] = {"full_names": list({person_name}), "sessions": [], "total_events": 0}
    else:
        names = set(data["people"][prefix].get("full_names", []))
        names.add(person_name)
        data["people"][prefix]["full_names"] = sorted(names)

    clip_count = _count_clips(target_folder)
    session = {
        "scenario": scenario_base,
        "folder": os.path.relpath(target_folder, os.path.dirname(os.path.abspath(__file__))),
        "youtube_link": youtube_link,
        "events": int(events_count),
        "clips": int(clip_count),
        "timestamp": datetime.utcnow().isoformat(),
    }
    data["people"][prefix]["sessions"].append(session)
    data["people"][prefix]["total_events"] = int(data["people"][prefix].get("total_events", 0)) + int(events_count)

    _save_progress(data)


def get_progress_summary():
    data = _load_progress()
    out = {}
    for prefix, rec in data.get("people", {}).items():
        out[prefix] = {"sessions": len(rec.get("sessions", [])), "total_events": int(rec.get("total_events", 0))}
    return out
