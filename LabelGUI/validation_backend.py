import os
import re
import cv2
import shutil
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
# Labeling sessions
# -----------------------
# One LabelingSession per video being labeled, keyed by its sid. The browser
# keeps its sid in the Flask session, so several people can label different
# videos on one server at the same time without touching each other's video,
# marks, pause or clock.

STREAM_MAX_WIDTH = 960   # frames sent to the browser; clips are cut from the original
FINISHED_SESSION_TTL_SEC = 3600

_sessions = {}
_sessions_lock = threading.Lock()
_progress_lock = threading.Lock()

CLIP_BEFORE_SEC = 2.0
CLIP_AFTER_SEC = 1.0


class LabelingSession:
    def __init__(self, sid, youtube_link=""):
        self.sid = sid
        self.lock = threading.Lock()
        self.link = (youtube_link or "").strip()
        self.state = "downloading"     # downloading | ready | failed | idle (cancelled)
        self.error = ""
        self.detail = ""
        self.video_file = None
        self.video_done = False
        self.duration = 0.0
        self.events = []               # [(idx, event_type, time_sec)]
        self.done_event = threading.Event()
        self.cancel_event = threading.Event()
        self.playback = video_utils.PlaybackState()
        self.stream_generation = 0
        self.extraction = {"in_progress": False, "current": 0, "total": 0}
        self.finished_at = None

    def end(self, cancel=False):
        with self.lock:
            if cancel:
                self.cancel_event.set()
            self.video_done = True
            self.done_event.set()
        self.playback.set_pause_flag(False)


def get_session(sid):
    with _sessions_lock:
        return _sessions.get(sid) if sid else None


def _register(session):
    now = time.time()
    with _sessions_lock:
        # Forget sessions that ended a while ago; their results are in the DB.
        for old_sid, old in list(_sessions.items()):
            if old.finished_at and now - old.finished_at > FINISHED_SESSION_TTL_SEC:
                del _sessions[old_sid]
        _sessions[session.sid] = session


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
    scenario_base=None,
    local_video_path=None,
    sid=None,
):
    """
    Runs one validation session: download, then wait for the labeler to finish,
    then cut the event clips. Blocks while the video downloads; use
    start_validation_async from a request handler.

    local_video_path: an uploaded video file to label instead of downloading
    youtube_link. It is never deleted, even with delete_original.

    Saves to:
        LabelGUI/ValidationResults/<folder_name>/
    """
    session = get_session(sid)
    if session is None:
        session = LabelingSession(sid or str(uuid.uuid4()), youtube_link)
        _register(session)
    sid = session.sid

    base_dir = os.path.dirname(os.path.abspath(__file__))

    youtube_downloads_dir = os.path.join(base_dir, "YouTubeDownloads")
    os.makedirs(youtube_downloads_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, "ValidationResults")
    os.makedirs(results_dir, exist_ok=True)

    # Build clean flat output folder
    folder_name = (folder_name or "").strip()

    if not folder_name:
        if person_name:
            folder_name = person_name
        else:
            folder_name = "Session"

    folder_name = sanitize_folder_name(folder_name)
    target_folder = get_unique_folder_name(results_dir, folder_name)

    clips_folder = os.path.join(target_folder, "clips")
    os.makedirs(clips_folder, exist_ok=True)

    if local_video_path:
        # Uploaded files are the only copy - never delete them after labeling.
        delete_original = False

    log_file_path = os.path.join(target_folder, "event_log.txt")

    db.upsert_validation_session(
        sid=sid,
        person_name=(person_name or ""),
        scenario_base=(scenario_base or ""),
        youtube_link=session.link,
        folder_path=os.path.relpath(target_folder, start=str(REPO_DIR)),
        delete_original=1 if bool(delete_original) else 0,
        status="running",
        duration_sec=0.0,
        events_count=0,
    )

    metadata_path = os.path.join(target_folder, "metadata.json")
    metadata = {
        "sid": sid,
        "folder_name": os.path.basename(target_folder),
        "person_name": person_name or "",
        "scenario_base": scenario_base or "",
        "youtube_link": session.link,
        "created_at": datetime.utcnow().isoformat(),
        "status": "running",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    if local_video_path:
        if os.path.isfile(local_video_path):
            downloaded_filepath, dl_error, dl_detail = local_video_path, "", ""
        else:
            downloaded_filepath = None
            dl_error = "The uploaded video file is missing from this computer."
            dl_detail = local_video_path
    else:
        downloaded_filepath, dl_error, dl_detail = download_video(
            youtube_link, youtube_downloads_dir
        )

    if session.cancel_event.is_set():
        with session.lock:
            session.state = "idle"
            session.finished_at = time.time()
        db.finalize_validation_session(sid, 0.0, 0, status="cancelled")
        return

    if not downloaded_filepath:
        with session.lock:
            session.video_done = True
            session.state = "failed"
            session.error = dl_error or "Download failed."
            session.detail = dl_detail or ""
            session.finished_at = time.time()
            session.done_event.set()

        db.finalize_validation_session(sid, 0.0, 0, status="failed")

        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("Download failed.\n")
            f.write(f"Link: {youtube_link}\n")
            f.write(f"Reason: {dl_error}\n")
            if dl_detail:
                f.write(f"Detail: {dl_detail}\n")

        metadata["status"] = "failed"
        metadata["error"] = dl_error
        metadata["error_detail"] = dl_detail
        metadata["finished_at"] = datetime.utcnow().isoformat()

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return

    with session.lock:
        session.video_file = downloaded_filepath
        session.state = "ready"

    db.upsert_validation_session(
        sid=sid,
        youtube_link=session.link,
        video_path=os.path.relpath(downloaded_filepath, start=str(REPO_DIR)),
        status="running",
    )

    def video_thread():
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(f"YouTube Link: {youtube_link}\n")
            f.write(f"Output Folder: {os.path.relpath(target_folder, base_dir)}\n")
            f.write(f"Clips Folder: {os.path.relpath(clips_folder, base_dir)}\n")
            f.write(f"Person: {person_name or ''}\n")
            f.write(f"Scenario: {scenario_base or ''}\n\n")

        # Wait until the labeler clicks Done (or the session is cancelled).
        session.done_event.wait()

        # Where playback stopped: the end of the video, or earlier when the
        # labeler finished early. Background windows only use [0, this].
        watched_until_sec = float(session.playback.get_current_time_sec() or 0.0)

        if session.cancel_event.is_set():
            db.finalize_validation_session(
                sid,
                duration_sec=float(session.duration or 0.0),
                events_count=int(len(session.events)),
                status="cancelled"
            )

            metadata["status"] = "cancelled"
            metadata["finished_at"] = datetime.utcnow().isoformat()
            metadata["events_count"] = int(len(session.events))

            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            session.finished_at = time.time()
            return

        with session.lock:
            events_snapshot = list(session.events)

        cap_for_fps = cv2.VideoCapture(downloaded_filepath)
        fps = (cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0) if cap_for_fps.isOpened() else 30.0
        cap_for_fps.release()

        multiple_pass_extract(
            downloaded_filepath,
            target_folder,
            events_snapshot,
            fps,
            log_file_path=log_file_path,
            progress=session.extraction,
        )

        if bool(delete_original) and os.path.exists(downloaded_filepath):
            try:
                os.remove(downloaded_filepath)
            except Exception:
                pass

        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(f"\nTotal Events Observed: {len(events_snapshot)}\n")

        try:
            update_progress_record(
                person_name=person_name,
                youtube_link=youtube_link,
                scenario_base=scenario_base,
                target_folder=target_folder,
                events_count=len(events_snapshot),
            )
        except Exception:
            pass

        duration_now = float(session.duration or 0.0)
        if duration_now > 0:
            watched_until_sec = min(watched_until_sec, duration_now)
        db.finalize_validation_session(
            sid,
            duration_sec=duration_now,
            events_count=int(len(events_snapshot)),
            status="final",
            watched_until_sec=watched_until_sec,
        )

        metadata["status"] = "final"
        metadata["finished_at"] = datetime.utcnow().isoformat()
        metadata["events_count"] = int(len(events_snapshot))
        metadata["duration_sec"] = duration_now
        metadata["watched_until_sec"] = watched_until_sec

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        session.finished_at = time.time()

    threading.Thread(target=video_thread, daemon=True).start()


def start_validation_async(**kwargs):
    """
    Start a validation session in the background and return its sid at once,
    so the page can show "Getting the video…" instead of hanging on the click.
    The session exists (state "downloading") before this returns.
    """
    session = LabelingSession(str(uuid.uuid4()), kwargs.get("youtube_link", ""))
    _register(session)
    threading.Thread(
        target=start_validation_thread, kwargs=dict(kwargs, sid=session.sid), daemon=True
    ).start()
    return session.sid


def cancel_validation(sid):
    """Abandon a session without saving clips (Skip, or the labeler started another video)."""
    session = get_session(sid)
    if session is not None:
        session.end(cancel=True)


def generate_video_stream(sid):
    """
    Streams the session's video via MJPEG frames.

    At the last frame the stream holds that frame instead of ending, so the
    labeler can still mark a landing or crash right at the end, or rewind.
    The session only ends when they click Done (finish_validation_early).
    """
    session = get_session(sid)
    if session is None or not session.video_file or not os.path.exists(session.video_file):
        return

    cap = cv2.VideoCapture(session.video_file)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    session.duration = (total_frames / fps) if total_frames else 0.0
    playback = session.playback
    playback.set_video_duration(session.duration)

    # Each connection gets a number; an older connection of the same session
    # stops as soon as a newer one opens, so a reconnect never runs two
    # players against one clock.
    with session.lock:
        session.stream_generation += 1
        my_generation = session.stream_generation

    if playback.get_current_time_sec() > 0:
        # Browser reconnected - carry on where it was.
        cap.set(cv2.CAP_PROP_POS_MSEC, playback.get_current_time_sec() * 1000)

    def should_stop():
        return session.stream_generation != my_generation or session.done_event.is_set()

    def draw_overlay(frame, current_time_sec):
        elapsed = video_utils.format_time(current_time_sec)
        total = video_utils.format_time(session.duration)
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

    for mjpeg_frame in video_utils.read_video_frames(
            cap, fps, draw_overlay, hold_at_end=True, should_stop=should_stop,
            state=playback, max_width=STREAM_MAX_WIDTH):
        if not mjpeg_frame:
            time.sleep(0.05)
            continue
        yield mjpeg_frame

    cap.release()
    if should_stop():
        return

    # Not a single frame could be decoded - end the session as before.
    session.end()


def _event_dict(idx, event_type, time_sec):
    return {"index": idx, "type": event_type, "time_sec": round(float(time_sec), 2)}


def mark_event_now(sid, event_type: str, reviewer: str = ""):
    """
    Called by /mark_event endpoint. reviewer = the logged-in user.
    Returns the saved event, or None when there is no video to mark.
    """
    session = get_session(sid)
    if session is None:
        return None
    current_time_sec = session.playback.get_current_time_sec()

    with session.lock:
        if session.state != "ready" or session.video_done or session.cancel_event.is_set():
            return None

        idx = (session.events[-1][0] + 1) if session.events else 1
        try:
            db.insert_validation_event(sid, idx, event_type, current_time_sec, reviewer=reviewer)
        except Exception as exc:
            print("[mark_event_now] could not save event:", exc)
            return None
        session.events.append((idx, event_type, current_time_sec))
        return _event_dict(idx, event_type, current_time_sec)


def undo_last_event(sid):
    """Remove the most recent mark of the session. Returns it, or None."""
    session = get_session(sid)
    if session is None:
        return None
    with session.lock:
        if session.video_done or not session.events:
            return None
        idx, event_type, time_sec = session.events[-1]
        try:
            db.delete_validation_event(sid, idx)
        except Exception as exc:
            print("[undo_last_event] could not delete event:", exc)
            return None
        session.events.pop()
        return _event_dict(idx, event_type, time_sec)


def list_current_events(sid):
    session = get_session(sid)
    if session is None:
        return []
    with session.lock:
        return [_event_dict(*e) for e in session.events]


def is_video_done(sid):
    session = get_session(sid)
    return True if session is None else session.video_done


def finish_validation_early(sid):
    """Stop playback and let the extraction thread save the marked events."""
    session = get_session(sid)
    if session is None or session.state != "ready":
        return False
    session.end()
    return True


def get_crash_count(sid):
    session = get_session(sid)
    return len(session.events) if session else 0


def get_extraction_progress(sid):
    session = get_session(sid)
    return dict(session.extraction) if session else {"in_progress": False, "current": 0, "total": 0}


def toggle_pause(sid):
    session = get_session(sid)
    return session.playback.toggle_pause_flag() if session else False


def skip_video(sid, offset_seconds: float):
    session = get_session(sid)
    if session is not None:
        session.playback.schedule_skip(offset_seconds)


# -----------------------
# Extraction (NO DUPLICATES)
# -----------------------
def multiple_pass_extract(video_path, target_folder, event_times_list, fps_hint, log_file_path=None,
                          progress=None):
    """Cut one clip per event. `progress` (a dict) is updated for the UI."""
    if not event_times_list:
        return

    sorted_times = sorted(event_times_list, key=lambda x: x[2])
    progress = progress if progress is not None else {}
    progress.update(in_progress=True, current=0, total=len(sorted_times))

    clips_folder = os.path.join(target_folder, "clips")
    os.makedirs(clips_folder, exist_ok=True)

    log_path = log_file_path or os.path.join(target_folder, "event_log.txt")

    excel_rows = []

    with open(log_path, "a", encoding="utf-8") as lf:
        for (idx, event_type, ctime) in sorted_times:
            progress["current"] += 1

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or fps_hint or 30.0
            event_frame = int(ctime * fps)

            start_frame = max(0, event_frame - int(CLIP_BEFORE_SEC * fps))
            end_frame = event_frame + int(CLIP_AFTER_SEC * fps)

            approx_start = sec_to_hms(start_frame / fps)
            approx_end = sec_to_hms(end_frame / fps)

            lf.write(
                f"{event_type} #{idx}: "
                f"event_time={round(ctime, 3)} sec, "
                f"frames={start_frame}-{end_frame}, "
                f"time={approx_start}-{approx_end}\n"
            )

            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            clip_filename = f"{idx:03d}_{event_type}.mp4"
            out_filename = os.path.join(clips_folder, clip_filename)

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

                # Write clean frame. No label text burned into the clip.
                writer.write(frame)

            if writer:
                writer.release()

            cap.release()

            excel_rows.append({
                "event_index": idx,
                "event_type": event_type,
                "event_time_sec": round(ctime, 3),
                "event_time_hms": sec_to_hms(ctime),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "approx_start_time": approx_start,
                "approx_end_time": approx_end,
                "clip_filename": clip_filename,
                "clip_path": os.path.join("clips", clip_filename),
            })

    progress["in_progress"] = False

    if excel_rows:
        csv_path = os.path.join(target_folder, "events.csv")
        excel_path = os.path.join(target_folder, "labels.xlsx")

        try:
            df = pd.DataFrame(excel_rows)
            df.to_csv(csv_path, index=False)
            df.to_excel(excel_path, index=False)
        except Exception as e:
            print("[multiple_pass_extract] Failed to write label files:", e)

# -----------------------
# Helpers
# -----------------------
def sec_to_hms(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def get_unique_folder_name(parent_dir, base_name):
    """Create and return a new folder: base_name, base_name1, base_name2, ...

    Created here (not just checked) so two labelers starting at the same moment
    never get the same folder."""
    counter = 0
    while True:
        candidate = os.path.join(parent_dir, base_name if counter == 0 else f"{base_name}{counter}")
        try:
            os.makedirs(candidate)
            return candidate
        except FileExistsError:
            counter += 1

def sanitize_folder_name(name: str) -> str:
    """
    Makes a safe Windows-friendly folder name.
    """
    name = (name or "").strip()

    # Replace invalid Windows folder characters
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)

    # Replace whitespace with underscore
    name = re.sub(r"\s+", "_", name)

    # Remove repeated underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing dots, underscores, and spaces
    name = name.strip("._ ")

    if not name:
        name = datetime.utcnow().strftime("Session_%Y%m%d_%H%M%S")

    return name


# -----------------------
# Download (clean)
# -----------------------
_FFMPEG_SHIM_DIR = BASE_DIR / ".ffmpeg_bin"
_ffmpeg_dir_cache = None


def _shim_imageio_ffmpeg(exe: str):
    """
    imageio-ffmpeg ships its binary under a platform-specific name
    (e.g. ffmpeg-macos-aarch64-v7.1). yt-dlp looks for a file literally named
    "ffmpeg" inside ffmpeg_location, so expose one via a symlink.
    """
    try:
        _FFMPEG_SHIM_DIR.mkdir(parents=True, exist_ok=True)
        link = _FFMPEG_SHIM_DIR / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

        if link.exists() or link.is_symlink():
            if link.is_symlink() and os.path.realpath(link) == os.path.realpath(exe):
                return str(_FFMPEG_SHIM_DIR)
            link.unlink()

        if os.name == "nt":
            shutil.copy2(exe, link)
        else:
            link.symlink_to(exe)

        os.chmod(exe, 0o755)
        return str(_FFMPEG_SHIM_DIR)
    except Exception as e:
        print("[ffmpeg] could not create shim:", e)
        return None


def _find_ffmpeg_dir():
    """
    Locate an ffmpeg binary in a portable way.

    Order: PATH -> imageio-ffmpeg's bundled static binary -> None.
    Returns a directory containing an executable named "ffmpeg", or None.
    """
    global _ffmpeg_dir_cache

    if _ffmpeg_dir_cache is not None:
        return _ffmpeg_dir_cache or None

    exe = shutil.which("ffmpeg")
    if exe:
        _ffmpeg_dir_cache = os.path.dirname(exe)
        return _ffmpeg_dir_cache

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            shim = _shim_imageio_ffmpeg(exe)
            if shim:
                _ffmpeg_dir_cache = shim
                return shim
    except Exception:
        pass

    _ffmpeg_dir_cache = ""
    return None


def has_ffmpeg():
    return _find_ffmpeg_dir() is not None


def _friendly_download_error(raw: str, youtube_link: str) -> str:
    """
    Turn a raw yt-dlp/network error into something a labeler can act on.
    """
    text = (raw or "").lower()

    if "certificate_verify_failed" in text or "certificate verify failed" in text:
        return (
            "Could not verify HTTPS certificates. This Python install has no CA "
            "certificate bundle. Fix it by running:  \"/Applications/Python 3.14/"
            "Install Certificates.command\"  (or: pip install --upgrade certifi)."
        )

    if "sign in to confirm" in text or "bot" in text and "confirm" in text:
        return (
            "YouTube asked for sign-in / bot confirmation for this video. "
            "It cannot be downloaded anonymously."
        )

    if "private video" in text:
        return "This video is private and cannot be downloaded."

    if ("video unavailable" in text
            or "not available" in text
            or "removed by the uploader" in text
            or "has been terminated" in text):
        return (
            "This video is no longer available on YouTube (removed, private, or "
            "region-blocked). Update the link in the Excel sheet."
        )

    if "age" in text and "restrict" in text:
        return "This video is age-restricted and cannot be downloaded anonymously."

    if "unsupported url" in text or "no video formats" in text:
        return f"No downloadable video found at this link: {youtube_link}"

    if "ffmpeg" in text:
        return (
            "ffmpeg is required to merge the video and audio streams but was not "
            "found. Install it with 'brew install ffmpeg' or "
            "'pip install imageio-ffmpeg'."
        )

    if "timed out" in text or "timeout" in text or "connection" in text:
        return "Network problem while downloading. Check your internet connection."

    return "Download failed. See details below."


def _precheck_link(youtube_link: str):
    """
    Reject links yt-dlp can never handle, before wasting time on attempts.
    Returns a friendly error string, or None if the link looks downloadable.
    """
    url = (youtube_link or "").strip()

    if not url:
        return "No video link was provided for this row."

    if not url.lower().startswith(("http://", "https://")):
        return f"This does not look like a valid URL: {url}"

    if "drive.google.com" in url.lower() and "/folders/" in url.lower():
        return (
            "This link is a Google Drive FOLDER, not a video file. Replace it in "
            "the Excel sheet with a direct link to a single video."
        )

    return None


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def download_video(youtube_link, download_folder):
    """
    Download the video behind `youtube_link` into `download_folder`.

    Returns (filepath, error_message, error_detail).
    On success error_message and error_detail are empty strings.
    """
    os.makedirs(download_folder, exist_ok=True)

    precheck = _precheck_link(youtube_link)
    if precheck:
        print(f"[download_video] rejected link: {precheck}")
        return None, precheck, youtube_link

    youtube_link = _normalize_youtube_url(youtube_link)

    ffmpeg_dir = _find_ffmpeg_dir()

    base_opts = {
        "outtmpl": os.path.join(download_folder, "%(title).50s-%(id)s.%(ext)s"),
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 5,
        "concurrent_fragment_downloads": 4,
    }

    if ffmpeg_dir:
        base_opts["ffmpeg_location"] = ffmpeg_dir

    if ffmpeg_dir:
        # ffmpeg available: best quality first, merging separate A/V streams.
        strategies = [
            "bv*+ba/bestvideo*+bestaudio",
            "best[ext=mp4]",
            "18",
            "best",
        ]
    else:
        # No ffmpeg: only pre-muxed (single-file) formats can work.
        strategies = [
            "best[ext=mp4][acodec!=none][vcodec!=none]",
            "18",
            "best[acodec!=none][vcodec!=none]",
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

    errors = []

    def try_with(fmt, player_clients=None):
        opts = dict(base_opts)
        opts["format"] = fmt
        if player_clients:
            opts["extractor_args"] = {"youtube": {"player_client": player_clients}}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_link, download=True)
                out_paths = _resolve_output_paths(ydl, info)
                for p in out_paths:
                    if p.lower().endswith((".mp4", ".mkv", ".webm")) and os.path.exists(p):
                        return p
            errors.append(f"[{fmt}] produced no playable file.")
        except Exception as e:
            errors.append(f"[{fmt}] {e}")
            print(f"[download_video] attempt with '{fmt}' failed:", e)
        return None

    for fmt in strategies:
        p = try_with(fmt)
        if p:
            return p, "", ""

    # YouTube often answers the default client with 403 / "format not
    # available" for videos that are fine; the mobile clients still get them.
    if "youtube.com" in youtube_link or "youtu.be" in youtube_link:
        for fmt in ("best[ext=mp4]", "best"):
            p = try_with(fmt, player_clients=["android", "web", "ios"])
            if p:
                return p, "", ""

    detail = _ANSI_RE.sub("", "\n".join(errors))

    # "Requested format is not available" from a fallback is a symptom, not the
    # cause - diagnose from the most specific error across all attempts.
    diagnostic = next(
        (e for e in errors if "ffmpeg" in e.lower()),
        None,
    ) or next(
        (e for e in errors if "requested format is not available" not in e.lower()),
        None,
    ) or detail

    if not ffmpeg_dir:
        diagnostic += " (ffmpeg was not found on this machine)"

    return None, _friendly_download_error(diagnostic, youtube_link), detail


def get_validation_status(sid):
    """
    Status of one labeling session, for the UI to poll.
    """
    session = get_session(sid)
    if session is None:
        return {"sid": sid, "state": "idle", "error": "", "detail": "", "video_ready": False,
                "video_name": "", "link": "", "done": True, "ffmpeg": bool(_find_ffmpeg_dir()),
                "can_remove_unavailable": False, "link_is_dead": False}
    with session.lock:
        error = session.error
        return {
            "sid": sid,
            "state": session.state,
            "error": error,
            "detail": session.detail,
            "video_ready": bool(session.video_file and os.path.exists(session.video_file)),
            "video_name": os.path.basename(session.video_file) if session.video_file else "",
            "link": session.link,
            "done": bool(session.video_done),
            "ffmpeg": bool(_find_ffmpeg_dir()),
            # Any failed video can be removed (the labeler confirms first);
            # this one tells the page whether the link is known to be dead.
            "can_remove_unavailable": session.state == "failed",
            "link_is_dead": session.state == "failed" and (
                "no longer available" in error.lower()
                or "private and cannot" in error.lower()
                or "no downloadable video" in error.lower()
            ),
        }


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

    with _progress_lock:   # several labelers can finish at the same time
        _update_progress(person_name, youtube_link, scenario_base, target_folder, events_count)


def _update_progress(person_name, youtube_link, scenario_base, target_folder, events_count):
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


