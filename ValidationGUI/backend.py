import os
import cv2
import yt_dlp
import threading
import time
from datetime import timedelta

# Globals
_processing_thread = None
_video_done = False
_crash_count = 0
_log_file_path = None
_delete_original = False
_crash_times = []     # [(crash_idx, timestamp_sec), ...]
_crash_flag = False
_video_duration = 0.0

# We'll store the "current frame time" from generate_video_stream
_current_frame_time_global = 0.0

def start_processing_thread(youtube_link, folder_name, delete_original):
    global _processing_thread
    global _video_done
    global _crash_count
    global _log_file_path
    global _delete_original
    global _crash_flag
    global _crash_times
    global _video_duration

    # If already processing in a thread, do nothing
    if _processing_thread and _processing_thread.is_alive():
        return

    # Reset globals
    _video_done = False
    _crash_count = 0
    _crash_flag = False
    _crash_times = []
    _video_duration = 0.0
    _delete_original = delete_original

    # Make folders
    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, 'YouTubeDownloads')
    os.makedirs(youtube_downloads_dir, exist_ok=True)

    results_dir = os.path.join(base_dir, 'Results')
    os.makedirs(results_dir, exist_ok=True)

    target_folder = get_unique_folder_name(results_dir, folder_name)
    os.makedirs(target_folder, exist_ok=True)

    _log_file_path = os.path.join(target_folder, "crash_log.txt")

    # 1) Download the video
    downloaded_filepath = download_video(youtube_link, youtube_downloads_dir)
    if not downloaded_filepath:
        # If fail, mark done
        _video_done = True
        with open(_log_file_path, 'w') as f:
            f.write("Download failed.\n")
        return

    def video_thread():
        """
        1) Wait for _video_done
        2) multiple-pass each crash
        3) write final logs
        """
        nonlocal downloaded_filepath, target_folder, youtube_link

        # Write initial info
        with open(_log_file_path, 'w') as f:
            f.write(f"YouTube Link: {youtube_link}\n")
            f.write(f"Folder: {os.path.basename(target_folder)}\n\n")

        # Wait until the streaming ends
        while not _video_done:
            time.sleep(0.5)

        # Grab FPS for rewriting clips
        cap_for_fps = cv2.VideoCapture(downloaded_filepath)
        if cap_for_fps.isOpened():
            fps = cap_for_fps.get(cv2.CAP_PROP_FPS) or 30.0
            cap_for_fps.release()
        else:
            fps = 30.0

        # multi-pass extraction
        multiple_pass_extract(downloaded_filepath, target_folder, _crash_times, fps)

        # If user wants to delete original
        if _delete_original and os.path.exists(downloaded_filepath):
            os.remove(downloaded_filepath)

        # Write final crash count
        with open(_log_file_path, 'a') as f:
            f.write(f"\nTotal Crashes Observed: {_crash_count}\n")

        finalize_video(target_folder, _crash_count)

    # Start background thread
    _processing_thread = threading.Thread(target=video_thread, daemon=True)
    _processing_thread.start()

def generate_video_stream():
    """
    Reads frames from the video in real-time. Overlays Elapsed/Total.
    If user clicks crash, we store that time. Once end is reached => _video_done = True
    """
    global _video_done, _video_duration
    global _crash_flag, _crash_times, _crash_count, _current_frame_time_global

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, 'YouTubeDownloads')

    # Find MP4
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

    frame_interval_sec = 1.0 / fps

    while True:
        ret, frame = cap.read()
        if not ret:
            # end of video
            cap.release()
            _video_done = True
            break

        current_frame_idx = cap.get(cv2.CAP_PROP_POS_FRAMES)
        current_time_sec = current_frame_idx / fps

        _current_frame_time_global = current_time_sec

        # If user clicked crash
        if _crash_flag:
            _crash_flag = False
            next_crash_idx = _crash_count + 1
            _crash_times.append((next_crash_idx, current_time_sec))
            _crash_count += 1

        # Overlay
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

        # MJPEG
        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + encodedImage.tobytes()
               + b'\r\n')

        time.sleep(frame_interval_sec)

def multiple_pass_extract(video_path, target_folder, crash_times_list, fps):
    """
    For each crash, open the video, read from (crash_time-10) to (crash_time+10),
    write to crash_{idx}.mp4
    """
    if not crash_times_list:
        return

    sorted_times = sorted(crash_times_list, key=lambda x: x[1])
    with open(_log_file_path, 'a') as lf:
        for (idx, ctime) in sorted_times:
            start_sec = max(0, ctime - 10)
            end_sec = ctime + 10
            lf.write(f"Crash #{idx}: [{sec_to_hms(start_sec)} - {sec_to_hms(end_sec)}]\n")

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                continue

            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            out_filename = os.path.join(target_folder, f"crash_{idx:02d}.mp4")
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

                # Overlay
                overlay_text = f"Crash #{idx}, T={sec_to_hms(current_time_sec)}"
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

def set_crash_flag():
    global _crash_flag
    _crash_flag = True

def is_video_done():
    return _video_done

def finalize_video(target_folder, crash_count):
    global _video_done
    _video_done = True

def get_crash_count():
    return _crash_count

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
