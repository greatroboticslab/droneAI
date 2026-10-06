"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
Needs the ML packages (requirements-ml.txt); skipped without scikit-learn.
"""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HAVE_SKLEARN = importlib.util.find_spec("sklearn") is not None

if HAVE_SKLEARN:
    import numpy as np  # noqa: E402
    import pandas as pd  # noqa: E402

    from train_dpflow_tabular_models import (  # noqa: E402
        attach_video_ids,
        load_excluded_videos,
        make_group_folds,
    )


def _clips(sessions_and_labels):
    rows = []
    for i, (session, label) in enumerate(sessions_and_labels):
        rows.append({"clip_group": f"c{i}", "session_name": session, "clip_filename": f"{i}.mp4",
                     "label": label, "video_id": "", "view_type": "", "f1": float(i)})
    return pd.DataFrame(rows)


@unittest.skipUnless(HAVE_SKLEARN, "scikit-learn not installed")
class GroupedCVTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.map_path = Path(self.tmp.name) / "session_videos.csv"
        pd.DataFrame([
            # Drone_1 and Drone_11 are the same video labeled twice.
            {"session_name": "Drone_1", "video_id": "VID_A", "view_type": "FPV"},
            {"session_name": "Drone_11", "video_id": "VID_A", "view_type": "FPV"},
            {"session_name": "Other", "video_id": "VID_B", "view_type": "Third-person"},
        ]).to_csv(self.map_path, index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_map_gives_same_video_to_relabeled_sessions(self):
        df, info = attach_video_ids(_clips([("Drone_1", "land"), ("Drone_11", "land"),
                                            ("Other", "takeoff"), ("Unmapped", "land")]), self.map_path)
        self.assertEqual(df["video_id"].tolist(), ["VID_A", "VID_A", "VID_B", "session:Unmapped"])
        self.assertEqual(df["view_type"].tolist(), ["FPV", "FPV", "Third-person", "Unknown"])
        self.assertEqual(info["clips_video_id_from_map"], 3)
        self.assertEqual(info["clips_grouped_by_session_fallback"], 1)

    def test_video_id_column_in_features_wins(self):
        clips = _clips([("Drone_1", "land")])
        clips.loc[0, "video_id"] = "FROM_FEATURES"
        df, info = attach_video_ids(clips, self.map_path)
        self.assertEqual(df.loc[0, "video_id"], "FROM_FEATURES")
        self.assertEqual(info["clips_video_id_from_features"], 1)

    def test_no_map_file_falls_back_to_session(self):
        df, info = attach_video_ids(_clips([("S1", "land")]), Path(self.tmp.name) / "missing.csv")
        self.assertEqual(df.loc[0, "video_id"], "session:S1")
        self.assertEqual(info["video_map"], "")

    def test_folds_never_split_a_video(self):
        rng = np.random.default_rng(0)
        rows = []
        for v in range(12):
            for c in range(int(rng.integers(2, 6))):
                rows.append((f"sess{v}", ["takeoff", "land", "minor-crash", "severe-crash"][(v + c) % 4]))
        df = _clips(rows)
        df["video_id"] = df["session_name"]
        folds, fold_of = make_group_folds(df, 5, seed=1)
        self.assertEqual(len(folds), 5)
        self.assertTrue((fold_of >= 0).all())
        per_video = df.assign(fold=fold_of).groupby("video_id")["fold"].nunique()
        self.assertTrue((per_video == 1).all())
        for tr, va in folds:
            self.assertFalse(set(df.loc[tr, "video_id"]) & set(df.loc[va, "video_id"]))

    def test_fewer_videos_than_folds_uses_fewer_folds(self):
        df = _clips([("a", "land"), ("b", "land"), ("b", "land"), ("c", "takeoff"), ("d", "land")])
        df["video_id"] = df["session_name"]
        folds, _ = make_group_folds(df, 5, seed=0)
        self.assertEqual(len(folds), 4)  # 4 videos

    def test_small_classes_use_fewer_folds(self):
        df = _clips([("a", "land"), ("b", "land"), ("c", "takeoff"), ("d", "takeoff"), ("e", "land")])
        df["video_id"] = df["session_name"]
        folds, _ = make_group_folds(df, 5, seed=0)
        self.assertEqual(len(folds), 3)  # largest class has 3 clips

    def test_one_video_is_an_error(self):
        df = _clips([("a", "land"), ("a", "takeoff")])
        df["video_id"] = "a"
        with self.assertRaises(ValueError):
            make_group_folds(df, 5, seed=0)

    def test_excluded_videos(self):
        p = Path(self.tmp.name) / "test_videos.csv"
        pd.DataFrame({"video_id": ["VID_A", "VID_C"]}).to_csv(p, index=False)
        self.assertEqual(load_excluded_videos(p), {"VID_A", "VID_C"})
        self.assertEqual(load_excluded_videos(Path(self.tmp.name) / "none.csv"), set())


if __name__ == "__main__":
    unittest.main()
