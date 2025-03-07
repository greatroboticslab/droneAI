import cv2
import time
from datetime import timedelta

_pause_flag = False
_pending_skip_offset = 0.0
_current_frame_time_global = 0.0
_video_duration_global = 0.0


def toggle_pause_flag():
    """
    Toggles the global pause state.
    """
    global _pause_flag
    _pause_flag = not _pause_flag
    return _pause_flag


def schedule_skip(offset_seconds: float):
    """
    Schedules a skip by `offset_seconds` 
    (negative => rewind, positive => fast-forward).
    """
    global _pending_skip_offset
    _pending_skip_offset += offset_seconds


def get_pause_flag():
    return _pause_flag


def get_and_clear_skip_offset():
    """
    Returns the current skip offset and resets it to 0.
    """
    global _pending_skip_offset
    val = _pending_skip_offset
    _pending_skip_offset = 0.0
    return val


def set_current_time_sec(sec: float):
    global _current_frame_time_global
    _current_frame_time_global = sec


def get_current_time_sec():
    return _current_frame_time_global


def set_video_duration(dur: float):
    global _video_duration_global
    _video_duration_global = dur


def get_video_duration():
    return _video_duration_global


def read_video_frames(video_capture, fps, frame_handler_callback=None):
    """
    A utility generator that continuously reads frames from `video_capture`.
    - `fps`: frames per second
    - `frame_handler_callback(frame, current_time_sec)` is optional and can
       be used to do additional logic (like overlay text, labeling color, etc.)
    
    Yields the MJPEG bytes for each frame or an empty byte if paused/no frame.
    """
    frame_interval_sec = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_encoded_frame = None

    while True:
        skip_val = get_and_clear_skip_offset()
        if skip_val != 0.0:
            new_time = get_current_time_sec() + skip_val
            if new_time < 0:
                new_time = 0
            if new_time > get_video_duration():
                new_time = get_video_duration()
            video_capture.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
            set_current_time_sec(new_time)

        if get_pause_flag():
            if last_encoded_frame is not None:
                yield last_encoded_frame
            time.sleep(0.1)
            continue

        ret, frame = video_capture.read()
        if not ret:
            video_capture.release()
            break

        current_frame_idx = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        current_time_sec = current_frame_idx / fps
        set_current_time_sec(current_time_sec)

        if frame_handler_callback:
            frame = frame_handler_callback(frame, current_time_sec)

        (flag, encodedImage) = cv2.imencode(".jpg", frame)
        if not flag:
            continue

        last_encoded_frame = (b'--frame\r\n'
                              b'Content-Type: image/jpeg\r\n\r\n'
                              + encodedImage.tobytes()
                              + b'\r\n')

        yield last_encoded_frame
        time.sleep(frame_interval_sec)


def format_time(seconds: float):
    return str(timedelta(seconds=int(seconds)))


def set_pause_flag(value: bool):
    """
    Set the global pause state to a specific value.
    """
    global _pause_flag
    _pause_flag = value
