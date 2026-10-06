import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import data_snapshot as ds  # noqa: E402
from db.db_store import DBStore  # noqa: E402


class DataSnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.db_path = self.root / "db.sqlite"
        db = DBStore(str(self.db_path))
        db.save_dataset("ds", "Flights", "f.xlsx", b"", "tester", is_active=True)
        db.replace_dataset_items("ds", [
            {"item_key": "ds:1", "row_index": 1, "person_name": "Ann",
             "youtube_link": "https://youtu.be/aaaaaaaaaaa", "status": "labeled", "view_type": "FPV"},
            {"item_key": "ds:2", "row_index": 2, "person_name": "Bob",
             "youtube_link": "https://youtu.be/bbbbbbbbbbb", "status": "not_labeled"},
        ])
        db.upsert_validation_session(sid="s1", youtube_link="https://youtu.be/aaaaaaaaaaa",
                                     folder_path="LabelGUI/ValidationResults/Ann", status="running")
        db.insert_validation_event("s1", 1, "takeoff", 3.0)
        db.insert_validation_event("s1", 2, "land", 9.5)
        db.finalize_validation_session("s1", 12.0, 2)
        self.db = db

        self.results = self.root / "ValidationResults"
        clips = self.results / "Ann" / "clips"
        clips.mkdir(parents=True)
        (clips / "001_takeoff.mp4").write_bytes(b"clip one")
        (clips / "002_land.mp4").write_bytes(b"clip two")
        (self.results / "Bob" / "clips").mkdir(parents=True)     # not labeled: left out
        (self.results / "Bob" / "clips" / "001_takeoff.mp4").write_bytes(b"other")

        self.test_split = self.root / "test_videos.csv"
        self.test_split.write_text("video_id\nzzzzzzzzzzz\n")
        self.out = self.root / "snapshots"

    def make(self, **kw):
        return ds.create_snapshot(self.db_path, self.results, self.test_split, self.out, **kw)

    def test_snapshot_holds_labeled_clips_db_and_split(self):
        snap = self.make()
        info = ds.read_snapshot(snap)
        self.assertEqual(info["id"], "v1")
        self.assertEqual((info["labeled_videos"], info["clips"]), (1, 2))
        self.assertEqual(info["events"]["takeoff"], 1)
        self.assertEqual(info["events"]["land"], 1)
        self.assertTrue((snap / "ValidationResults/Ann/clips/002_land.mp4").exists())
        self.assertFalse((snap / "ValidationResults/Bob").exists())
        for name in ("droneai.sqlite", "session_videos.csv", "test_videos.csv", "video_manifest.csv"):
            self.assertTrue((snap / name).exists(), name)
        with zipfile.ZipFile(snap.with_suffix(".zip")) as zf:
            self.assertIn(f"{snap.name}/snapshot.json", zf.namelist())

    def test_same_data_same_fingerprint_new_label_new_fingerprint(self):
        first = ds.read_snapshot(self.make(make_zip=False))
        second = ds.read_snapshot(self.make(make_zip=False))
        self.assertEqual(second["id"], "v2")
        self.assertEqual(first["fingerprint"], second["fingerprint"])
        self.db.insert_validation_event("s1", 3, "minor-crash", 10.0)
        third = ds.read_snapshot(self.make(make_zip=False))
        self.assertNotEqual(first["fingerprint"], third["fingerprint"])
        self.assertEqual([s["id"] for s in ds.list_snapshots(self.out)], ["v3", "v2", "v1"])

    def test_fingerprint_detects_a_changed_clip(self):
        snap = self.make(make_zip=False)
        stored = json.loads((snap / "snapshot.json").read_text())["fingerprint"]
        self.assertEqual(ds.fingerprint(snap), stored)
        (snap / "ValidationResults/Ann/clips/001_takeoff.mp4").write_bytes(b"edited")
        self.assertNotEqual(ds.fingerprint(snap), stored)

    def test_needs_a_test_split(self):
        self.test_split.unlink()
        with self.assertRaises(FileNotFoundError):
            self.make()


if __name__ == "__main__":
    unittest.main()
