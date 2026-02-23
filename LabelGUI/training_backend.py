import os
import cv2
import yt_dlp
import threading
import time
import csv
from datetime import timedelta
import json

import video_utils

import uuid
from pathlib import Path
from db.db_store import DBStore

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"
db = DBStore(str(DB_PATH))

_training_sid = None

###############################################################################
#                     GLOBALS for the Training Workflow                       #
###############################################################################
_training_thread = None

_delete_original = False
_training_video_path = None
_training_metadata_path = None
_training_done = False

_user_name = ""
_labels_and_colors = []
_active_label = None
_capture_mode = "10fps"
_custom_fps = 1.0

_label_chunks = []
_last_frame_index = 0

_final_pass_in_progress = False
_final_pass_current = 0
_final_pass_total = 0

_delete_metadata_after_final = True

_saved_frames_count = {}

###############################################################################
#              PUBLIC API: Start / Resume / Generate Streams                  #
###############################################################################
def start_training_session(
    youtube_link,
    user_name,
    delete_original,
    capture_mode,
    custom_fps,
    labels_and_colors,
    keep_metadata=False
):
    global _training_thread
    global _training_done, _training_video_path, _training_metadata_path
    global _delete_original, _user_name, _labels_and_colors, _active_label
    global _capture_mode, _custom_fps, _label_chunks, _last_frame_index
    global _delete_metadata_after_final

    global _training_sid

    _training_sid = str(uuid.uuid4())

    db.upsert_training_session(
    sid=_training_sid,
    user_name=user_name,
    youtube_link=youtube_link,
    video_path=str(_training_video_path or ""),
    metadata_path=str(_training_metadata_path or ""),
    capture_mode=str(capture_mode),
    custom_fps=float(custom_fps),
    delete_original=1 if delete_original else 0,
    keep_metadata=1 if keep_metadata else 0,
    status="running",
)

    if _training_thread and _training_thread.is_alive():
        return

    _training_done = False
    _label_chunks.clear()
    _last_frame_index = 0
    _active_label = None

    _delete_original = delete_original
    _user_name = user_name
    _labels_and_colors = labels_and_colors
    _capture_mode = capture_mode
    _custom_fps = custom_fps
    _delete_metadata_after_final = (not keep_metadata)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    youtube_downloads_dir = os.path.join(base_dir, 'YouTubeDownloads')
    os.makedirs(youtube_downloads_dir, exist_ok=True)

    training_dir = os.path.join(base_dir, 'TrainingResults')
    os.makedirs(training_dir, exist_ok=True)

    meta_folder = os.path.join(training_dir, 'MetaDataLocation')
    os.makedirs(meta_folder, exist_ok=True)

    _training_metadata_path = os.path.join(meta_folder, "metadata.csv")

    _training_video_path = download_video(youtube_link, youtube_downloads_dir)
    if not _training_video_path:
        _training_done = True
        return

    def background_thread():
        while not _training_done:
            time.sleep(0.5)

    _training_thread = threading.Thread(target=background_thread, daemon=True)
    _training_thread.start()


def resume_training_session(metadata_file):
    global _training_done, _training_video_path, _training_metadata_path
    global _label_chunks, _active_label, _user_name, _capture_mode, _custom_fps
    global _labels_and_colors, _last_frame_index, _delete_metadata_after_final

    _training_done = False
    _training_metadata_path = metadata_file

    if not os.path.exists(metadata_file):
        return

    _label_chunks.clear()
    with open(metadata_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        mode = "header"
        for row in reader:
            if len(row) < 2:
                continue
            if mode == "header":
                key, val = row[0], row[1]
                if key == "user_name":
                    _user_name = val
                elif key == "video_path":
                    _training_video_path = val
                elif key == "capture_mode":
                    _capture_mode = val
                elif key == "custom_fps":
                    _custom_fps = float(val)
                elif key == "active_label":
                    _active_label = val if val else None
                elif key == "labels":
                    _labels_and_colors = []
                    label_entries = val.split(';')
                    for entry in label_entries:
                        if '|' in entry:
                            lbl, color = entry.split('|', 1)
                            _labels_and_colors.append((lbl, color))
                elif key == "last_frame_index":
                    _last_frame_index = int(val)
                elif key == "delete_metadata_after_final":
                    _delete_metadata_after_final = (val.strip() == "1")
                elif key == "----":
                    mode = "chunks"
            elif mode == "chunks":
                c_start, c_end, c_label = row
                _label_chunks.append({
                    "start_frame": int(c_start),
                    "end_frame": int(c_end),
                    "label": c_label
                })

    from video_utils import set_pause_flag
    set_pause_flag(False)

    def background_thread():
        while not _training_done:
            time.sleep(0.5)

    t = threading.Thread(target=background_thread, daemon=True)
    t.start()


def generate_training_preview_stream():
    global _training_video_path, _training_done

    if not _training_video_path or not os.path.exists(_training_video_path):
        return b''

    cap = cv2.VideoCapture(_training_video_path)
    if not cap.isOpened():
        return b''

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    preview_duration = 10.0

    def overlay_preview(frame, current_time_sec):
        h, w, _ = frame.shape
        cv2.putText(
            frame,
            "Preview Mode (10s)",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 255, 255),
            3
        )
        return frame

    start_time = time.time()
    for mjpeg_frame in video_utils.read_video_frames(cap, fps, frame_handler_callback=overlay_preview):
        if not mjpeg_frame:
            time.sleep(0.1)
            continue

        elapsed = time.time() - start_time
        if elapsed > preview_duration:
            break

        yield mjpeg_frame

    cap.release()


def generate_training_video_stream(auto_finalize=True):
    global _training_done, _active_label, _last_frame_index

    if not _training_video_path or not os.path.exists(_training_video_path):
        _training_done = True
        while not _training_done:
            time.sleep(0.1)
        return b''

    cap = cv2.VideoCapture(_training_video_path)
    if not cap.isOpened():
        _training_done = True
        return b''

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0
    video_utils.set_video_duration(duration_sec)

    if _last_frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, _last_frame_index)
        video_utils.set_current_time_sec(_last_frame_index / fps)
    else:
        video_utils.set_current_time_sec(0.0)

    def overlay_label(frame, current_time_sec):
        label_color = (255, 255, 255)
        if _active_label:
            for (lbl, hx) in _labels_and_colors:
                if lbl == _active_label:
                    rgb = hex_to_bgr(hx)
                    label_color = rgb
                    break
        h, w, _ = frame.shape
        cv2.rectangle(frame, (0, 0), (w, h), label_color, 10)
        overlay_text = f"Label: {_active_label or 'None'}"
        cv2.putText(
            frame,
            overlay_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        return frame

    for mjpeg_frame in video_utils.read_video_frames(cap, fps, frame_handler_callback=overlay_label):
        if not mjpeg_frame:
            time.sleep(0.1)
            continue

        current_frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        if _active_label and (current_frame_idx - 1) >= _last_frame_index:
            add_or_update_chunk(_active_label, _last_frame_index, current_frame_idx - 1)
            _last_frame_index = current_frame_idx

        yield mjpeg_frame

    if not _training_done:
        _training_done = True
        if auto_finalize:
            finalize_training_session(do_final_pass=True)


###############################################################################
#                            CHUNK MANAGEMENT                                 #
###############################################################################
def add_or_update_chunk(label, start_frame, end_frame):
    if not _label_chunks:
        _label_chunks.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "label": label
        })
        return

    last_chunk = _label_chunks[-1]
    if last_chunk["label"] == label and last_chunk["end_frame"] + 1 >= start_frame:
        last_chunk["end_frame"] = end_frame
    else:
        _label_chunks.append({
            "start_frame": start_frame,
            "end_frame": end_frame,
            "label": label
        })


###############################################################################
#               FINAL PASS: Write Labeled Frames to Disk                      #
###############################################################################
def finalize_training_session(do_final_pass=True):
    global _training_done, _final_pass_in_progress
    global _final_pass_current, _final_pass_total, _saved_frames_count

    print(f"\nDEBUG: finalize_training_session called with do_final_pass={do_final_pass}")
    print("DEBUG: current chunks:", _label_chunks)

    _training_done = True

    save_metadata_csv(_training_metadata_path)

    if not do_final_pass:
        print("DEBUG: do_final_pass=False -> skipping image writes.")
        return

    _final_pass_in_progress = True
    _saved_frames_count.clear()

    total_frames_to_save = 0
    for chunk in _label_chunks:
        chunk_len = chunk["end_frame"] - chunk["start_frame"] + 1
        total_frames_to_save += chunk_len

    _final_pass_total = total_frames_to_save
    _final_pass_current = 0
    print(f"DEBUG: total_frames_to_save={total_frames_to_save}")

    if _training_video_path and os.path.exists(_training_video_path):
        cap = cv2.VideoCapture(_training_video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_idx = 0

            base_dir = os.path.dirname(os.path.abspath(__file__))
            training_dir = os.path.join(base_dir, 'TrainingResults')
            os.makedirs(training_dir, exist_ok=True)

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                assigned_label = find_label_for_frame(frame_idx)
                if assigned_label:
                    if should_save_this_frame(frame_idx, fps):
                        success = save_frame_to_label_folder_debug(frame, assigned_label, frame_idx, training_dir)
                        if success:
                            _final_pass_current += 1
                frame_idx += 1

            cap.release()

        if _delete_original and os.path.exists(_training_video_path):
            os.remove(_training_video_path)
            print("DEBUG: original video deleted")

    _final_pass_in_progress = False

    if _delete_metadata_after_final and _training_metadata_path and os.path.exists(_training_metadata_path):
        os.remove(_training_metadata_path)
        print("DEBUG: metadata file removed -> final pass complete")


def save_frame_to_label_folder_debug(frame, label, frame_idx, training_dir):
    global _user_name, _saved_frames_count

    label_folder = os.path.join(training_dir, label)
    os.makedirs(label_folder, exist_ok=True)

    safe_user_name = make_safe_name(_user_name)
    base_filename = f"{safe_user_name}_{label}_{frame_idx}.png"
    final_filename = find_non_collision_filename(label_folder, base_filename)
    out_path = os.path.join(label_folder, final_filename)
    success = cv2.imwrite(out_path, frame)

    if success:
        if label not in _saved_frames_count:
            _saved_frames_count[label] = 0
        _saved_frames_count[label] += 1
    else:
        print(f"DEBUG: cv2.imwrite failed for {out_path}")

    return success


def find_label_for_frame(frame_idx: int):
    for chunk in _label_chunks:
        if chunk["start_frame"] <= frame_idx <= chunk["end_frame"]:
            return chunk["label"]
    return None


def should_save_this_frame(frame_idx: int, fps: float):
    global _capture_mode, _custom_fps
    if _capture_mode == "all":
        return True
    elif _capture_mode == "10fps":
        divisor = int(round(fps / 10.0))
        if divisor < 1:
            divisor = 1
        return (frame_idx % divisor) == 0
    elif _capture_mode == "customfps":
        if _custom_fps <= 0:
            return False
        divisor = int(round(fps / _custom_fps))
        if divisor < 1:
            divisor = 1
        return (frame_idx % divisor) == 0
    else:
        # fallback to 10fps
        divisor = int(round(fps / 10.0))
        if divisor < 1:
            divisor = 1
        return (frame_idx % divisor) == 0


###############################################################################
#                            SET ACTIVE LABEL                                 #
###############################################################################
def set_current_label(label_name):
    global _active_label
    print(f"DEBUG: set_current_label -> {label_name}")
    _active_label = label_name


###############################################################################
#                                GET STATUS                                   #
###############################################################################
def is_training_done():
    return _training_done


def get_training_status():
    total_frames = 0
    for c in _label_chunks:
        total_frames += (c["end_frame"] - c["start_frame"] + 1)

    total_saved = sum(_saved_frames_count.values())

    return {
        "user_name": _user_name,
        "total_chunks": len(_label_chunks),
        "total_labeled_frames": total_frames,
        "label_chunks": _label_chunks,
        "saved_frames_per_label": _saved_frames_count,
        "saved_frames_total": total_saved
    }


def get_training_progress():
    return {
        "final_pass_in_progress": _final_pass_in_progress,
        "final_pass_current": _final_pass_current,
        "final_pass_total": _final_pass_total
    }


###############################################################################
#                           METADATA CSV SAVE/LOAD                            #
###############################################################################
def save_metadata_csv(path):
    if not path:
        return
    print(f"DEBUG: save_metadata_csv -> {path}")
    rows = []
    rows.append(["user_name", _user_name])
    rows.append(["video_path", _training_video_path or ""])
    rows.append(["capture_mode", _capture_mode])
    rows.append(["custom_fps", str(_custom_fps)])
    rows.append(["active_label", _active_label or ""])
    rows.append(["session_id", _training_sid or ""])

    label_str_parts = []
    for (lbl, col) in _labels_and_colors:
        label_str_parts.append(f"{lbl}|{col}")
    label_str = ";".join(label_str_parts)
    rows.append(["labels", label_str])
    rows.append(["last_frame_index", str(_last_frame_index)])

    from training_backend import _delete_metadata_after_final
    rows.append(["delete_metadata_after_final", "1" if _delete_metadata_after_final else "0"])

    rows.append(["----", "chunks"])
    for c in _label_chunks:
        rows.append([c["start_frame"], c["end_frame"], c["label"]])

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(rows)
    print("DEBUG: wrote metadata CSV, chunk rows:", len(_label_chunks))


###############################################################################
#                          LABEL GROUP LOAD/SAVE                              #
###############################################################################
def load_label_group_file(path):
    """
    Reads a .lblgroup JSON file with structure:
    {
      "group_name": "MyDroneLabels",
      "labels": [
         {"label": "Crash", "color": "#FF0000"},
         ...
      ]
    }
    Returns a list of (label, color).
    """
    if not os.path.exists(path):
        print(f"DEBUG: label group file missing: {path}")
        return []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        pairs = []
        for item in data.get("labels", []):
            pairs.append((item["label"], item["color"]))
        return pairs
    except Exception as e:
        print(f"DEBUG: load_label_group_file error: {e}")
        return []


def save_label_group_file(path, labels_and_colors):
    """
    Writes a .lblgroup JSON file with structure:
    {
      "group_name": "filename without extension",
      "labels": [
         {"label": "Crash", "color": "#FF0000"},
         ...
      ]
    }
    """
    base = os.path.splitext(os.path.basename(path))[0]
    data = {
        "group_name": base,
        "labels": []
    }
    for (lbl, col) in labels_and_colors:
        data["labels"].append({"label": lbl, "color": col})

    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"DEBUG: saved label group -> {path}")
    except Exception as e:
        print(f"DEBUG: save_label_group_file error: {e}")


###############################################################################
#                          HELPER FUNCTIONS                                   #
###############################################################################
def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 6:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)
    return (255, 255, 255)


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
    except Exception as e:
        print(f"DEBUG: download_video exception: {e}")
    return None


def get_current_labels():
    return _labels_and_colors


def make_safe_name(name):
    return name.replace(" ", "_")


def find_non_collision_filename(folder_path, filename):
    base, ext = os.path.splitext(filename)
    if not os.path.exists(os.path.join(folder_path, filename)):
        return filename
    count = 2
    while True:
        new_filename = f"{base}({count}){ext}"
        if not os.path.exists(os.path.join(folder_path, new_filename)):
            return new_filename
        count += 1
