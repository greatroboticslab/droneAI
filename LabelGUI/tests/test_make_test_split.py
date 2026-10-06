import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_test_split import pick_test_videos  # noqa: E402


def manifest(rows):
    cols = ["video_id", "view_type", "status", "n_takeoff", "n_land", "n_minor_crash", "n_severe_crash"]
    return pd.DataFrame(rows, columns=cols).assign(source="Real", person_name="x")


class TestSplitTests(unittest.TestCase):
    def setUp(self):
        rows = [(f"fpv{i}", "FPV", "labeled", 1, 0, 0, 0) for i in range(12)]
        rows += [(f"tp{i}", "Third-person", "labeled", 1, 0, 0, 0) for i in range(4)]
        rows += [
            ("land1", "Third-person", "labeled", 0, 1, 0, 0),
            ("minor1", "FPV", "labeled", 0, 0, 2, 0),
            ("sev1", "FPV", "labeled", 0, 0, 0, 1),
            ("empty", "FPV", "labeled", 0, 0, 0, 0),
            ("todo", "FPV", "not_labeled", 3, 3, 3, 3),
        ]
        self.m = manifest(rows)

    def test_twenty_percent_of_labeled_videos_with_events(self):
        test = pick_test_videos(self.m)
        self.assertEqual(len(test), 4)  # 19 eligible videos -> 4
        self.assertNotIn("empty", set(test.video_id))
        self.assertNotIn("todo", set(test.video_id))

    def test_every_event_class_is_covered(self):
        test = pick_test_videos(self.m)
        for col in ["n_takeoff", "n_land", "n_minor_crash", "n_severe_crash"]:
            self.assertGreater(test[col].sum(), 0, col)

    def test_same_seed_same_split(self):
        self.assertEqual(list(pick_test_videos(self.m).video_id),
                         list(pick_test_videos(self.m).video_id))


if __name__ == "__main__":
    unittest.main()
