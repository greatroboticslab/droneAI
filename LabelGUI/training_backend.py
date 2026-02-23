import os
import cv2
import yt_dlp
import threading
import time
import csv
from datetime import timedelta
import json

import uuid
from pathlib import Path
from db.db_store import DBStore

BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = BASE_DIR.parent
DB_PATH = REPO_DIR / "db" / "droneai.sqlite"
db = DBStore(str(DB_PATH))

_training_sid = None

import video_utils

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

    # ---------------- DB: create a session ----------------
    _training_sid = str(uuid.uuid4())
    db.upsert_training_session(
        sid=_training_sid,
        user_name=_user_name,
        youtube_link=youtube_link,
        video_path=_training_video_path or "",
        metadata_path=_training_metadata_path or "",
        capture_mode=_capture_mode,
        custom_fps=float(_custom_fps),
        delete_original=1 if _delete_original else 0,
        keep_metadata=0 if _delete_metadata_after_final else 1,
        status="running",
    )

    def training_thread_fn():
        # Save metadata initially so Resume works even if user leaves early
        save_metadata_csv(_training_metadata_path)

        # Training ends when generate_training_video_stream hits end of video
        # finalize_training_session will be called by generate_training_video_stream auto-finalize
        # or by the user via Take a Break route
        return

    _training_thread = threading.Thread(target=training_thread_fn, daemon=True)
    _training_thread.start()


def resume_training_session(metadata_file):
    global _training_done, _training_video_path, _training_metadata_path
    global _label_chunks, _active_label, _user_name, _capture_mode, _custom_fps
    global _labels_and_colors, _last_frame_index, _delete_metadata_after_final
    global _training_sid

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
                elif key == "session_id":
                    _training_sid = val
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
            else:
                # chunk rows
                if len(row) >= 3:
                    try:
                        s = int(row[0])
                        e = int(row[1])
                        lbl = row[2]
                        _label_chunks.append({"start_frame": s, "end_frame": e, "label": lbl})
                    except Exception:
                        pass

    # Mark session running in DB when resuming
    if _training_sid:
        db.upsert_training_session(
            sid=_training_sid,
            user_name=_user_name,
            video_path=_training_video_path or "",
            metadata_path=_training_metadata_path or "",
            capture_mode=_capture_mode,
            custom_fps=float(_custom_fps),
            delete_original=1 if _delete_original else 0,
            keep_metadata=0 if _delete_metadata_after_final else 1,
            status="running",
        )


def generate_training_preview_stream():
    # Quick preview mode: show the video from the beginning
    if not _training_video_path or not os.path.exists(_training_video_path):
        return

    cap = cv2.VideoCapture(_training_video_path)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # just show first ~3 seconds
    max_frames = int(fps * 3)
    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        ret2, buffer = cv2.imencode('.jpg', frame)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1
        time.sleep(1 / fps)

    cap.release()


def generate_training_video_stream(auto_finalize=True):
    global _training_done
    global _last_frame_index

    if not _training_video_path or not os.path.exists(_training_video_path):
        _training_done = True
        return

    cap = cv2.VideoCapture(_training_video_path)
    if not cap.isOpened():
        _training_done = True
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = (total_frames / fps) if total_frames else 0.0

    video_utils.set_video_duration(duration_sec)

    # Seek to last_frame_index if resuming
    if _last_frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, _last_frame_index)

    while True:
        if video_utils.is_paused():
            time.sleep(0.1)
            continue

        skip = video_utils.consume_skip_request()
        if skip != 0:
            cur = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            newpos = max(0, cur + int(skip * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, newpos)

        ret, frame = cap.read()
        if not ret:
            break

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        _last_frame_index = frame_idx
        video_utils.set_current_time_sec(frame_idx / fps)

        # show overlay: time and active label
        overlay = frame.copy()
        elapsed = video_utils.format_time(frame_idx / fps)
        total = video_utils.format_time(duration_sec)
        label_txt = _active_label or "(no label)"
        cv2.putText(overlay, f"{elapsed} / {total}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(overlay, f"Label: {label_txt}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        ret2, buffer = cv2.imencode('.jpg', overlay)
        if not ret2:
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(1 / fps)

    cap.release()

    if auto_finalize:
        finalize_training_session(do_final_pass=True)


###############################################################################
#                       CHUNKS + FINALIZATION LOGIC                           #
###############################################################################
def add_or_update_chunk(start_frame, end_frame, label):
    global _label_chunks

    if start_frame > end_frame:
        start_frame, end_frame = end_frame, start_frame

    # try merge with last chunk if same label and adjacent/overlapping
    if _label_chunks:
        last = _label_chunks[-1]
        if last["label"] == label and start_frame <= last["end_frame"] + 1:
            last["end_frame"] = max(last["end_frame"], end_frame)
            return

    _label_chunks.append({"start_frame": start_frame, "end_frame": end_frame, "label": label})


def finalize_training_session(do_final_pass=True):
    global _training_done, _final_pass_in_progress
    global _training_sid
    global _final_pass_current, _final_pass_total, _saved_frames_count

    print(f"\nDEBUG: finalize_training_session called with do_final_pass={do_final_pass}")
    print("DEBUG: current chunks:", _label_chunks)

    _training_done = True

    save_metadata_csv(_training_metadata_path)

    if not do_final_pass:
        print("DEBUG: do_final_pass=False -> skipping image writes.")
        if _training_sid:
            db.upsert_training_session(
                sid=_training_sid,
                user_name=_user_name,
                video_path=_training_video_path or "",
                metadata_path=_training_metadata_path or "",
                capture_mode=_capture_mode,
                custom_fps=float(_custom_fps),
                delete_original=1 if _delete_original else 0,
                keep_metadata=0 if _delete_metadata_after_final else 1,
                status="paused",
            )
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

    # Finalize DB session
    if _training_sid:
        db.replace_training_chunks(_training_sid, _label_chunks)
        db.upsert_training_session(
            sid=_training_sid,
            user_name=_user_name,
            video_path=_training_video_path or "",
            metadata_path=_training_metadata_path or "",
            capture_mode=_capture_mode,
            custom_fps=float(_custom_fps),
            delete_original=1 if _delete_original else 0,
            keep_metadata=0 if _delete_metadata_after_final else 1,
            status="final",
        )


def save_frame_to_label_folder_debug(frame, label, frame_idx, training_dir):
    safe_label = make_safe_name(label)
    label_dir = os.path.join(training_dir, safe_label)
    os.makedirs(label_dir, exist_ok=True)

    if safe_label not in _saved_frames_count:
        _saved_frames_count[safe_label] = 0
    _saved_frames_count[safe_label] += 1
    seq = _saved_frames_count[safe_label]

    filename = f"{safe_label}_{seq:06d}.jpg"
    out_path = os.path.join(label_dir, filename)

    try:
        cv2.imwrite(out_path, frame)
        return True
    except Exception as e:
        print("DEBUG: failed to write frame:", e)
        return False


def find_label_for_frame(frame_idx):
    for c in _label_chunks:
        if c["start_frame"] <= frame_idx <= c["end_frame"]:
            return c["label"]
    return None


def should_save_this_frame(frame_idx, fps):
    if _capture_mode == "10fps":
        return (frame_idx % max(1, int(fps / 10))) == 0
    if _capture_mode == "1fps":
        return (frame_idx % max(1, int(fps))) == 0
    if _capture_mode == "custom":
        # custom_fps frames per second
        step = max(1, int(fps / max(0.01, float(_custom_fps))))
        return (frame_idx % step) == 0
    return True


def set_current_label(label_name):
    global _active_label

    # close previous chunk if open-ended tracking is used somewhere else
    _active_label = label_name


###############################################################################
#                         STATUS / PROGRESS APIs                              #
###############################################################################
def is_training_done():
    return _training_done


def get_training_status():
    return {
        "training_done": _training_done,
        "final_pass_in_progress": _final_pass_in_progress,
        "final_pass_current": _final_pass_current,
        "final_pass_total": _final_pass_total
    }


def get_training_progress():
    if not _training_video_path or not os.path.exists(_training_video_path):
        return {"progress": 0.0}

    cap = cv2.VideoCapture(_training_video_path)
    if not cap.isOpened():
        return {"progress": 0.0}

    total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()

    if total <= 0:
        return {"progress": 0.0}

    return {"progress": float(_last_frame_index) / float(total)}


###############################################################################
#                            METADATA SAVE/LOAD                               #
###############################################################################
def save_metadata_csv(path):
    if not path:
        return
    print(f"DEBUG: save_metadata_csv -> {path}")
    rows = []
    rows.append(["user_name", _user_name])
    rows.append(["session_id", _training_sid or ""])
    rows.append(["video_path", _training_video_path or ""])
    rows.append(["capture_mode", _capture_mode])
    rows.append(["custom_fps", str(_custom_fps)])
    rows.append(["active_label", _active_label or ""])

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
def load_label_group_file(lblgroup_path):
    if not os.path.exists(lblgroup_path):
        return []
    lines = []
    with open(lblgroup_path, 'r', encoding='utf-8') as f:
        lines = [x.strip() for x in f.readlines() if x.strip()]
    out = []
    for ln in lines:
        parts = ln.split(',')
        if len(parts) >= 2:
            out.append((parts[0].strip(), parts[1].strip()))
    return out


def save_label_group_file(lblgroup_path, labels_and_colors):
    with open(lblgroup_path, 'w', encoding='utf-8') as f:
        for (lbl, col) in labels_and_colors:
            f.write(f"{lbl},{col}\n")


def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) != 6:
        return (255, 255, 255)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)


###############################################################################
#                                 DOWNLOAD                                    #
###############################################################################
def download_video(youtube_link, download_folder):
    os.makedirs(download_folder, exist_ok=True)

    # NOTE: keep your existing path; later we can make it config/env-based
    FFMPEG_DIR = r"C:\Users\rusha\Downloads\ffmpeg-8.0-essentials_build\ffmpeg-8.0-essentials_build\bin"

    ydl_opts = {
        'format': 'bv*+ba/bestvideo*+bestaudio',
        'outtmpl': os.path.join(download_folder, '%(title).50s-%(id)s.%(ext)s'),
        'merge_output_format': 'mp4',
        'noplaylist': True,
        'quiet': True,
        'no_warnings': True,
        'ffmpeg_location': FFMPEG_DIR,
        'extractor_args': {'youtube': {'player_client': ['android']}},
        'retries': 5,
        'concurrent_fragment_downloads': 4,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_link, download=True)
            # Try to resolve output filename
            try:
                p = ydl.prepare_filename(info)
                base, _ = os.path.splitext(p)
                mp4 = base + ".mp4"
                if os.path.exists(mp4):
                    return mp4
                if os.path.exists(p):
                    return p
            except Exception:
                pass
    except Exception as e:
        print("DEBUG: download_video failed:", e)

    return None


###############################################################################
#                           LABEL LIST for UI                                 #
###############################################################################
def get_current_labels():
    return _labels_and_colors


def make_safe_name(name: str) -> str:
    safe = []
    for ch in (name or ""):
        if ch.isalnum() or ch in ('-', '_'):
            safe.append(ch)
        else:
            safe.append('_')
    return "".join(safe).strip('_') or "label"


def find_non_collision_filename(folder, base_filename):
    candidate = os.path.join(folder, base_filename)
    if not os.path.exists(candidate):
        return candidate
    idx = 1
    while True:
        name, ext = os.path.splitext(base_filename)
        new_name = f"{name}_{idx}{ext}"
        candidate = os.path.join(folder, new_name)
        if not os.path.exists(candidate):
            return candidate
        idx += 1
