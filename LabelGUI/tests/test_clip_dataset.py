"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
The video-cutting tests need opencv (skipped without it).
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clip_dataset import (  # noqa: E402
    clip_window,
    find_video_file,
    load_labeled_sessions,
    recut_session,
)
from db.db_store import DBStore  # noqa: E402

HAVE_CV2 = importlib.util.find_spec("cv2") is not None


BAR_STEP = 4  # pixels per frame


def make_numbered_video(path, seconds=10, fps=10):
    """
    Black frames with a white bar at x = BAR_STEP * frame_index, so each frame's
    number can be read back exactly. (Brightness would not work: every mp4v
    re-encode darkens frames a little.)
    """
    import cv2
    import numpy as np

    n = seconds * fps
    width = BAR_STEP * n + 16
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, 32))
    for i in range(n):
        f = np.zeros((32, width, 3), dtype=np.uint8)
        f[:, BAR_STEP * i: BAR_STEP * i + BAR_STEP, :] = 255
        w.write(f)
    w.release()


def frame_numbers(path):
    """The frame number shown in each frame of a video made by make_numbered_video."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    numbers = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        cols = f.mean(axis=(0, 2))
        bright = (cols > 128).nonzero()[0]  # the bar, edges blurred by re-encoding
        center = float(bright.mean())
        numbers.append(int(round((center - (BAR_STEP - 1) / 2) / BAR_STEP)))
    cap.release()
    return numbers


def _item(key, row, link, **extra):
    base = {"item_key": f"{key}:{row}", "row_index": row, "person_name": "P", "youtube_link": link,
            "status": "labeled", "labeled_by": "x", "locked_by": "", "scenario_type": "",
            "video_id": "", "view_type": "FPV", "source": "Real", "local_path": ""}
    base.update(extra)
    return base


class ClipDatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.db_path = self.dir / "t.sqlite"
        self.video = self.dir / "Flight-AAAAAAAAAAA.mp4"
        db = DBStore(str(self.db_path))
        db.save_dataset("ds", "D", "d.xlsx", b"x", "leader", is_active=True)
        db.replace_dataset_items("ds", [_item("ds", 1, "https://youtu.be/AAAAAAAAAAA", video_id="AAAAAAAAAAA")])
        # Older finished session, then a newer one for the same video.
        db.upsert_validation_session(sid="old", youtube_link="https://youtu.be/AAAAAAAAAAA",
                                     folder_path="LabelGUI/ValidationResults/P_1", video_path=str(self.video),
                                     status="final", updated_at="2026-01-01T00:00:00")
        db.insert_validation_event("old", 1, "land", 4.0)
        db.upsert_validation_session(sid="new", youtube_link="https://www.youtube.com/watch?v=AAAAAAAAAAA",
                                     folder_path="LabelGUI/ValidationResults/P_11", video_path=str(self.video),
                                     status="running")
        for idx, ev, t in [(1, "takeoff", 1.0), (2, "minor-crash", 5.0), (3, "land", 9.5), (4, "oops", 6.0)]:
            db.insert_validation_event("new", idx, ev, t)
        db.finalize_validation_session("new", 10.0, 4)

    def tearDown(self):
        self.tmp.cleanup()

    def test_clip_window_is_clamped(self):
        self.assertEqual(clip_window(1.0, 2.0, 10.0, 60.0), (0.0, 11.0))
        self.assertEqual(clip_window(55.0, 2.0, 10.0, 60.0), (53.0, 60.0))
        self.assertEqual(clip_window(30.0, 2.0, 10.0, 0.0), (28.0, 40.0))  # unknown duration

    def test_newest_session_per_video(self):
        sessions = load_labeled_sessions(self.db_path)
        self.assertEqual([s["sid"] for s in sessions], ["new"])
        s = sessions[0]
        self.assertEqual(s["video_id"], "AAAAAAAAAAA")
        self.assertEqual(s["view_type"], "FPV")
        self.assertEqual(s["session_name"], "P_11")
        self.assertEqual([e[1] for e in s["events"]], ["takeoff", "minor-crash", "oops", "land"])  # by time

    def test_all_sessions(self):
        self.assertEqual({s["sid"] for s in load_labeled_sessions(self.db_path, all_sessions=True)}, {"old", "new"})

    def test_find_video_file_in_download_dir(self):
        dl = self.dir / "downloads"
        dl.mkdir()
        (dl / "Some title-AAAAAAAAAAA.mp4").write_bytes(b"x")
        s = {"local_path": "", "video_path": "gone/file.mp4", "video_id": "AAAAAAAAAAA"}
        self.assertEqual(find_video_file(s, download_dir=dl), dl / "Some title-AAAAAAAAAAA.mp4")
        self.assertIsNone(find_video_file({"local_path": "", "video_path": "", "video_id": "url-x"}, download_dir=dl))

    @unittest.skipUnless(HAVE_CV2, "opencv not installed")
    def test_recut_cuts_the_right_frames(self):
        make_numbered_video(self.video, seconds=10, fps=10)
        s = load_labeled_sessions(self.db_path)[0]
        self.assertEqual(find_video_file(s), self.video.resolve())
        out = self.dir / "out"
        rows, skipped = recut_session(s, self.video, out, before=2.0, after=3.0)
        self.assertEqual(skipped, 1)  # the "oops" event
        self.assertEqual([r["event_type"] for r in rows], ["takeoff", "minor-crash", "land"])

        crash = rows[1]
        self.assertEqual((crash["clip_start_sec"], crash["clip_end_sec"]), (3.0, 8.0))
        self.assertEqual(crash["frames"], 51)  # frames 30..80
        self.assertEqual(frame_numbers(self.video), list(range(100)))  # the reader works
        self.assertEqual(frame_numbers(out / "P_11" / "clips" / "002_minor-crash.mp4"), list(range(30, 81)))
        self.assertEqual(frame_numbers(out / "P_11" / "clips" / "001_takeoff.mp4"), list(range(0, 41)))

        land = rows[2]  # 9.5 s: window clamped at the 10 s end
        self.assertEqual((land["clip_start_sec"], land["clip_end_sec"]), (7.5, 10.0))
        self.assertEqual(land["video_id"], "AAAAAAAAAAA")

    def test_dry_run_writes_nothing(self):
        s = load_labeled_sessions(self.db_path)[0]
        out = self.dir / "dry"
        rows, _ = recut_session(s, None, out, before=2.0, after=10.0, dry_run=True)
        self.assertEqual(len(rows), 3)
        self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
