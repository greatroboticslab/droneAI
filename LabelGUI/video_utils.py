import cv2
import time
from datetime import timedelta

class PlaybackState:
    """
    Pause / skip / clock of one video player. The labeling page gives every
    session its own, so several people can label at once on one server; the
    module-level functions below use a shared default one (single-user pages).
    """

    def __init__(self):
        self.paused = False
        self.pending_skip = 0.0
        self.current_time = 0.0
        self.duration = 0.0

    def toggle_pause_flag(self):
        self.paused = not self.paused
        return self.paused

    def set_pause_flag(self, value: bool):
        self.paused = bool(value)

    def get_pause_flag(self):
        return self.paused

    def schedule_skip(self, offset_seconds: float):
        self.pending_skip += offset_seconds

    def get_and_clear_skip_offset(self):
        val, self.pending_skip = self.pending_skip, 0.0
        return val

    def set_current_time_sec(self, sec: float):
        self.current_time = sec

    def get_current_time_sec(self):
        return self.current_time

    def set_video_duration(self, dur: float):
        self.duration = dur

    def get_video_duration(self):
        return self.duration


_default = PlaybackState()


def toggle_pause_flag():
    """Toggles the global pause state."""
    return _default.toggle_pause_flag()


def schedule_skip(offset_seconds: float):
    """Schedules a skip by `offset_seconds` (negative => rewind, positive => fast-forward)."""
    _default.schedule_skip(offset_seconds)


def get_pause_flag():
    return _default.get_pause_flag()


def get_and_clear_skip_offset():
    """Returns the current skip offset and resets it to 0."""
    return _default.get_and_clear_skip_offset()


def set_current_time_sec(sec: float):
    _default.set_current_time_sec(sec)


def get_current_time_sec():
    return _default.get_current_time_sec()


def set_video_duration(dur: float):
    _default.set_video_duration(dur)


def get_video_duration():
    return _default.get_video_duration()


def read_video_frames(video_capture, fps, frame_handler_callback=None,
                      hold_at_end=False, should_stop=None, state=None, max_width=None):
    """
    A utility generator that continuously reads frames from `video_capture`.
    - `fps`: frames per second
    - `frame_handler_callback(frame, current_time_sec)` is optional and can
       be used to do additional logic (like overlay text, labeling color, etc.)
    - `hold_at_end`: at the last frame, keep showing it (and keep honouring
       rewinds) instead of ending, until `should_stop()` returns True.
    - `should_stop()`: optional; the generator ends as soon as it returns True.
    - `state`: the PlaybackState to use (default: the shared one).
    - `max_width`: shrink wider frames to this width before sending (lighter
       stream over a network; the saved clips are always cut from the original).

    Yields the MJPEG bytes for each frame or an empty byte if paused/no frame.
    """
    st = state or _default
    frame_interval_sec = 1.0 / fps if fps > 0 else 1.0 / 30.0
    last_encoded_frame = None

    at_end = False

    while True:
        if should_stop is not None and should_stop():
            break

        skip_val = st.get_and_clear_skip_offset()
        if skip_val != 0.0:
            new_time = st.get_current_time_sec() + skip_val
            if new_time < 0:
                new_time = 0
            if new_time > st.get_video_duration():
                new_time = st.get_video_duration()
            video_capture.set(cv2.CAP_PROP_POS_MSEC, new_time * 1000)
            st.set_current_time_sec(new_time)
            at_end = False

        if st.get_pause_flag() or at_end:
            if last_encoded_frame is not None:
                yield last_encoded_frame
            time.sleep(0.1)
            continue

        ret, frame = video_capture.read()
        if not ret:
            if hold_at_end and last_encoded_frame is not None:
                at_end = True
                continue
            video_capture.release()
            break

        current_frame_idx = video_capture.get(cv2.CAP_PROP_POS_FRAMES)
        current_time_sec = current_frame_idx / fps
        st.set_current_time_sec(current_time_sec)

        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)

        if frame_handler_callback:
            frame = frame_handler_callback(frame, current_time_sec)

        params = [cv2.IMWRITE_JPEG_QUALITY, 80] if max_width else []
        (flag, encodedImage) = cv2.imencode(".jpg", frame, params)
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
    """Set the global pause state to a specific value."""
    _default.set_pause_flag(value)
