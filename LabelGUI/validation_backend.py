import os
import cv2
import yt_dlp
import threading
import time
from datetime import timedelta
import video_utils

_processing_thread = None
_video_done = False
_log_file_path = None
_delete_original = False
_event_times = []     # Stores (idx, event_type, time_sec)
_video_duration = 0.0

_extraction_in_progress = False
_extraction_current = 0
_extraction_total = 0


def start_validation_thread(youtube_link, folder_name, delete_original):
    """
    Starts validation: downloads video, prepares log file, spawns processing thread.
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
        nonlocal downloaded_filepath, target_folder, youtube_link

        with open(_log_file_path, 'w') as f:
            f.write(f"YouTube Link: {youtube_link}\n")
            f.write(f"Folder: {os.path.basename(target_folder)}\n\n")

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

        finalize_video(target_folder)

    _processing_thread = threading.Thread(target=video_thread, daemon=True)
    _processing_thread.start()


def generate_video_stream():
    """
    Streams video frames with overlay (time info).
    """
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
        overlay_text = f"{elapsed_str} / {total_str}"
        cv2.putText(
            frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2
        )
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, draw_validation_overlay):
        if not mjpeg_frame:
            time.sleep(0.1)
            continue
        yield mjpeg_frame

    _video_done = True


def multiple_pass_extract(video_path, target_folder, event_times_list, fps):
    """
    Extracts short clips (5s: 2 before, 3 after) for each marked event.
    """
    global _extraction_in_progress, _extraction_current, _extraction_total, _log_file_path

    if not event_times_list:
        return

    sorted_times = sorted(event_times_list, key=lambda x: x[2])  # sort by time

    _extraction_in_progress = True
    _extraction_current = 0
    _extraction_total = len(sorted_times)

    with open(_log_file_path, 'a') as lf:
        for (idx, event_type, ctime) in sorted_times:
            _extraction_current += 1

            start_sec = max(0, ctime - 2)
            end_sec = ctime + 3
            lf.write(f"{event_type.capitalize()} #{idx}: [{sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}]\n")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            out_filename = os.path.join(target_folder, f"{event_type}_{idx:02d}.mp4")
            writer = None

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                if current_time_sec > end_sec:
                    break
                if current_time_sec < start_sec:
                    continue

                if writer is None:
                    h, w, _ = frame.shape
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_filename, fourcc, fps, (w, h))

                overlay_text = f"{event_type.capitalize()} #{idx}, T={sec_to_hms(current_time_sec)}"
                frame_copy = frame.copy()
                cv2.putText(
                    frame_copy,
                    overlay_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2
                )
                writer.write(frame_copy)

            if writer:
                writer.release()
            cap.release()

    _extraction_in_progress = False


def mark_event_now(event_type: str):
    """
    Records an event with its timestamp.
    """
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
    # returns number of events for compatibility
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
    ydl_opts = {
        'outtmpl': os.path.join(download_folder, '%(title).50s.%(ext)s'),
        'format': 'mp4/bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4'
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=True)
            if info.get('_filename'):
                return info['_filename']
            else:
                title = info.get('title', 'video')
                guessed_path = os.path.join(download_folder, f"{title[:50]}.mp4")
                if os.path.exists(guessed_path):
                    return guessed_path
    except Exception:
        pass
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
    """
    Returns a list of events in structured form for rendering in HTML.
    """
    results = []
    for (idx, event_type, ctime) in _event_times:
        start_sec = max(0, ctime - 2)
        end_sec = ctime + 3
        results.append({
            "index": idx,
            "type": event_type,
            "start": sec_to_hms(start_sec),
            "end": sec_to_hms(end_sec)
        })
    return results
