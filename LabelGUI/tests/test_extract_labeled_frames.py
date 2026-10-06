import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from extract_labeled_frames import select_sessions  # noqa: E402


class SelectSessionsTests(unittest.TestCase):
    manifest = [
        {"video_id": "a", "status": "labeled", "labeled_sid": "s2", "view_type": "FPV"},
        {"video_id": "b", "status": "labeled", "labeled_sid": "s3", "view_type": "Third-person"},
        {"video_id": "c", "status": "not_labeled", "labeled_sid": "", "view_type": "FPV"},
    ]
    sessions = [
        {"sid": "s1", "session_name": "Ann"},    # older session of video a: not used
        {"sid": "s2", "session_name": "Ann1"},
        {"sid": "s3", "session_name": "Bob"},
        {"sid": "s4", "session_name": "Cat"},    # unlabeled video
    ]

    def test_only_the_labeled_session_of_each_video(self):
        self.assertEqual(select_sessions(self.manifest, self.sessions), ["Ann1", "Bob"])

    def test_view_filter(self):
        self.assertEqual(select_sessions(self.manifest, self.sessions, "FPV"), ["Ann1"])


if __name__ == "__main__":
    unittest.main()
