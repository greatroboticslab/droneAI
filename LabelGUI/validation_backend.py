import os
import cv2
import yt_dlp
import threading
import time
from datetime import timedelta
import video_utils
import json
from datetime import datetime

_processing_thread = None
_video_done = False
_log_file_path = None
_delete_original = False
_event_times = []     # (idx, event_type, time_sec)
_video_duration = 0.0

_extraction_in_progress = False
_extraction_current = 0
_extraction_total = 0

import re

def _normalize_youtube_url(url: str) -> str:
    url = (url or "").strip().strip('"').strip("'")
    # youtu.be/<id>  -> https://www.youtube.com/watch?v=<id>
    m = re.match(r'^https?://youtu\.be/([A-Za-z0-9_-]{8,})', url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(1)}"
    # shorts -> watch
    m = re.match(r'^https?://(www\.)?youtube\.com/shorts/([A-Za-z0-9_-]{8,})', url)
    if m:
        return f"https://www.youtube.com/watch?v={m.group(2)}"
    return url



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
      ValidationResults/<unique folder_name/>
    """
    global _processing_thread, _video_done, _log_file_path
    global _delete_original, _event_times, _video_duration

    if _processing_thread and _processing_thread.is_alive():
        return

    _video_done = False
    _event_times = []
    _video_duration = 0.0
    _delete_original = delete_original

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, 'YouTubeDownloads')
    os.makedirs(youtube_downloads_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, 'ValidationResults')
    os.makedirs(results_dir, exist_ok=True)

    # target folder
    if person_name and scenario_base:
        prefix = (person_name or "").strip()[:4] or "User"
        person_dir = os.path.join(results_dir, prefix)
        os.makedirs(person_dir, exist_ok=True)
        existing = [d for d in os.listdir(person_dir) if os.path.isdir(os.path.join(person_dir, d))]
        count = sum(1 for d in existing if d.lower().startswith(scenario_base.lower()))
        scenario_dir_name = f"{scenario_base} {count + 1}"
        target_folder = os.path.join(person_dir, scenario_dir_name)
        os.makedirs(target_folder, exist_ok=True)
    else:
        if not folder_name:
            folder_name = "Session"
        target_folder = get_unique_folder_name(results_dir, folder_name)
        os.makedirs(target_folder, exist_ok=True)

    _log_file_path = os.path.join(target_folder, "event_log.txt")

    downloaded_filepath = download_video(youtube_link, youtube_downloads_dir)
    if not downloaded_filepath:
        _video_done = True
        with open(_log_file_path, 'w') as f:
            f.write("Download failed.\n")
        return

    def video_thread():
        nonlocal downloaded_filepath, target_folder, youtube_link, base_dir

        with open(_log_file_path, 'w') as f:
            f.write(f"YouTube Link: {youtube_link}\n")
            f.write(f"Folder: {os.path.relpath(target_folder, base_dir)}\n\n")

        while not _video_done:
            time.sleep(0.5)

        cap_for_fps = cv2.VideoCapture(downloaded_filepath)
        if cap_for_fps.isOpened():
            fps = cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0
            cap_for_fps.release()
        else:
            fps = 30.0

        multiple_pass_extract(downloaded_filepath, target_folder, _event_times, fps)

        if _delete_original and os.path.exists(downloaded_filepath):
            os.remove(downloaded_filepath)

        with open(_log_file_path, 'a') as f:
            f.write(f"\nTotal Events Observed: {len(_event_times)}\n")
        try:
            update_progress_record(
                person_name=person_name,
                youtube_link=youtube_link,
                scenario_base=scenario_base,
                target_folder=target_folder,
                events_count=len(_event_times)
            )
        except Exception:
            pass


        finalize_video(target_folder)

    _processing_thread = threading.Thread(target=video_thread, daemon=True)
    _processing_thread.start()


def generate_video_stream():
    global _video_done, _video_duration

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, 'YouTubeDownloads')

    files = sorted(
        [os.path.join(youtube_downloads_dir, f) for f in os.listdir(youtube_downloads_dir)],
        key=os.path.getmtime
    )
    video_file = None
    for f in reversed(files):
        if f.lower().endswith('.mp4'):
            video_file = f
            break

    if not video_file:
        while not _video_done:
            yield b''
            time.sleep(0.1)
        return

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        while not _video_done:
            yield b''
            time.sleep(0.1)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    if total_frames > 0:
        _video_duration = total_frames / fps
    else:
        _video_duration = 0.0

    video_utils.set_video_duration(_video_duration)
    video_utils.set_current_time_sec(0.0)

    def draw_validation_overlay(frame, current_time_sec):
        elapsed_str = str(timedelta(seconds=int(current_time_sec)))
        total_str = str(timedelta(seconds=int(_video_duration)))
        cv2.putText(frame, f"{elapsed_str} / {total_str}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, draw_validation_overlay):
        if not mjpeg_frame:
            time.sleep(0.1)
            continue
        yield mjpeg_frame

    _video_done = True


def multiple_pass_extract(video_path, target_folder, event_times_list, fps_unused):
    """
    Frame-based clips: 15 frames before + 15 after (~1s at 30fps).
    """
    global _extraction_in_progress, _extraction_current, _extraction_total, _log_file_path

    if not event_times_list:
        return

    sorted_times = sorted(event_times_list, key=lambda x: x[2])

    _extraction_in_progress = True
    _extraction_current = 0
    _extraction_total = len(sorted_times)

    with open(_log_file_path, 'a') as lf:
        for (idx, event_type, ctime) in sorted_times:
            _extraction_current += 1

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            event_frame = int(ctime * fps)
            frame_window = 15  # 15 before + 15 after

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
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_filename, fourcc, fps, (w, h))

                overlay_text = f"{event_type.capitalize()} #{idx}, Frame={current_frame_idx}"
                frame_copy = frame.copy()
                cv2.putText(frame_copy, overlay_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                writer.write(frame_copy)

            if writer:
                writer.release()
            cap.release()

    _extraction_in_progress = False


def mark_event_now(event_type: str):
    global _event_times
    idx = len(_event_times) + 1
    current_time_sec = video_utils.get_current_time_sec()
    _event_times.append((idx, event_type, current_time_sec))


def is_video_done():
    return _video_done


def finalize_video(target_folder):
    global _video_done
    _video_done = True


def get_crash_count():
    # return number of events (back-compat name)
    return len(_event_times)


def get_log_file_path():
    return _log_file_path


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


def download_video(youtube_link, download_folder):
    youtube_link = _normalize_youtube_url(youtube_link)
    os.makedirs(download_folder, exist_ok=True)

    FFMPEG_DIR = r"C:\Users\rusha\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

    ydl_opts = {
        "outtmpl": os.path.join(download_folder, "%(title).50s.%(ext)s"),
        # Try best video+audio; if separate, yt-dlp will merge with ffmpeg
        "format": "bestvideo*+bestaudio/best",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "ffmpeg_location": FFMPEG_DIR,
        # Helps with some recent YouTube player variants
        "extractor_args": {"youtube": {"player_client": ["web"]}},
        # Be tolerant
        "concurrent_fragment_downloads": 3,
        "retries": 5,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=True)
            out = info.get("_filename")
            if out and os.path.exists(out):
                print("[download_video] downloaded:", out)
                return out
            # fallback: guess common extensions by title
            title = (info.get("title") or "video")[:50]
            for ext in ("mp4", "mkv", "webm", "m4a"):
                cand = os.path.join(download_folder, f"{title}.{ext}")
                if os.path.exists(cand):
                    print("[download_video] downloaded (guessed):", cand)
                    return cand
    except Exception as e:
        print("[download_video] DOWNLOAD failed:", e)

    print("[download_video] failed to download for:", youtube_link)
    return None


    # Helper: pick best video format, then best audio. Prefer mp4, but accept any.
    def best_video(formats):
        vids = [f for f in formats if f.get("vcodec") not in (None, "none")]
        # prefer mp4 container
        mp4s = [f for f in vids if str(f.get("ext", "")).lower() == "mp4"]
        pool = mp4s if mp4s else vids
        # sort by approximate quality/bitrate
        pool.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        return pool[0] if pool else None

    def best_audio(formats):
        auds = [f for f in formats if f.get("acodec") not in (None, "none")]
        # prefer m4a for mp4 compatibility
        m4as = [f for f in auds if str(f.get("ext", "")).lower() in ("m4a", "mp4")]
        pool = m4as if m4as else auds
        pool.sort(key=lambda f: (f.get("abr") or 0, f.get("tbr") or 0), reverse=True)
        return pool[0] if pool else None

    # Single-file (already muxed) candidates (prefer mp4)
    muxed = [f for f in formats if f.get("vcodec") not in (None, "none") and f.get("acodec") not in (None, "none")]
    muxed_mp4 = [f for f in muxed if str(f.get("ext", "")).lower() == "mp4"]
    chosen_single = None
    if muxed_mp4:
        muxed_mp4.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        chosen_single = muxed_mp4[0]
    elif muxed:
        muxed.sort(key=lambda f: (f.get("height") or 0, f.get("tbr") or 0), reverse=True)
        chosen_single = muxed[0]

    v = best_video(formats)
    a = best_audio(formats)

    # Build format selector
    if chosen_single:
        fmt_selector = chosen_single.get("format_id")
    elif v and a:
        fmt_selector = f"{v.get('format_id')}+{a.get('format_id')}"
    elif v:
        fmt_selector = v.get("format_id")
    else:
        print("[download_video] No usable video stream (video formats missing).")
        return None

    # 2) Download with the chosen format selector
    dl_opts = dict(base_opts)
    dl_opts["format"] = fmt_selector

    try:
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            finfo = ydl.extract_info(youtube_link, download=True)
            # determine the actual output path
            out = finfo.get("_filename")
            if out and os.path.exists(out):
                print("[download_video] downloaded:", out)
                return out
            # guess by title
            title = finfo.get("title", "video")[:50]
            # check common extensions
            for ext in ("mp4", "mkv", "webm", "m4a"):
                cand = os.path.join(download_folder, f"{title}.{ext}")
                if os.path.exists(cand):
                    print("[download_video] downloaded (guessed):", cand)
                    return cand
    except Exception as e:
        print("[download_video] DOWNLOAD failed:", e)

    print("[download_video] failed to download for:", youtube_link)
    return None


    def try_download(opts):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(youtube_link, download=True)
                # Try to find the resulting filename
                # yt-dlp may return '_filename' or 'requested_downloads' or 'files'
                filename = info.get('_filename') or info.get('url')
                if filename and os.path.exists(filename):
                    return filename
                # fallback guess by title/ext
                title = info.get('title', 'video')
                for ext in ('mp4', 'mkv', 'webm', 'm4a', 'mp3'):
                    candidate = os.path.join(download_folder, f"{title[:50]}.{ext}")
                    if os.path.exists(candidate):
                        return candidate
        except Exception as e:
            print("[download_video] yt-dlp attempt failed:", e)
        return None

    # First try preferred mp4-friendly option
    path = try_download(ydl_opts_primary)
    if path:
        print("[download_video] primary succeeded:", path)
        return path

    # Then try the fallback permissive option
    path = try_download(ydl_opts_fallback)
    if path:
        print("[download_video] fallback succeeded:", path)
        return path

    print("[download_video] failed to download video for link:", repr(youtube_link))
    return None


def sec_to_hms(sec: float) -> str:
    return str(timedelta(seconds=int(sec)))


def get_extraction_progress():
    return {
        "in_progress": _extraction_in_progress,
        "current": _extraction_current,
        "total": _extraction_total
    }


def skip_video(offset_seconds: float):
    video_utils.schedule_skip(offset_seconds)


def toggle_pause():
    return video_utils.toggle_pause_flag()


def get_logged_events():
    # Light-weight preview for pages; exact window is logged in event_log.txt
    results = []
    for (idx, event_type, ctime) in _event_times:
        results.append({
            "index": idx,
            "type": event_type,
            "start": sec_to_hms(max(0, ctime - 1)),
            "end": sec_to_hms(ctime + 1)
        })
    return results

def _progress_path():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, 'ValidationResults')
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
    """Update progress.json after a session finishes."""
    if not person_name or not scenario_base:
        return  # only track Excel-driven sessions

    prefix = (person_name or "").strip()[:4] or "User"
    data = _load_progress()

    if "people" not in data:
        data["people"] = {}
    if prefix not in data["people"]:
        data["people"][prefix] = {
            "full_names": list({person_name}),
            "sessions": [],
            "total_events": 0
        }
    else:
        # remember full names we’ve seen for this prefix
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
        "timestamp": datetime.utcnow().isoformat()
    }
    data["people"][prefix]["sessions"].append(session)
    data["people"][prefix]["total_events"] = int(data["people"][prefix].get("total_events", 0)) + int(events_count)

    _save_progress(data)

def get_progress_summary():
    """Lightweight summary for UI: { prefix: {sessions:int, total_events:int} }"""
    data = _load_progress()
    out = {}
    for prefix, rec in data.get("people", {}).items():
        out[prefix] = {
            "sessions": len(rec.get("sessions", [])),
            "total_events": int(rec.get("total_events", 0))
        }
    return out

