"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
"""
import csv
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from db.db_store import DBStore  # noqa: E402
from video_manifest import COLUMNS, build_video_manifest, write_video_manifest  # noqa: E402


def _item(key, row, person, link, **extra):
    base = {
        "item_key": f"{key}:{row}", "row_index": row, "person_name": person,
        "youtube_link": link, "status": "not_labeled", "labeled_by": "", "locked_by": "",
        "scenario_type": "", "video_id": "", "view_type": "Unknown", "source": "Unknown",
        "local_path": "",
    }
    base.update(extra)
    return base


class VideoManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.sqlite"
        self.db = DBStore(str(self.db_path))

        self.db.save_dataset("ds1", "Phase 1", "p1.xlsx", b"x", "leader", is_active=True)
        self.db.replace_dataset_items("ds1", [
            _item("ds1", 1, "Alice", "https://youtu.be/AAAAAAAAAAA",
                  video_id="AAAAAAAAAAA", view_type="FPV", source="Real"),
            # Same video again, under another person: must be merged into one row.
            _item("ds1", 2, "Bob", "https://www.youtube.com/watch?v=AAAAAAAAAAA",
                  video_id="AAAAAAAAAAA", view_type="Unknown", source="Real"),
            _item("ds1", 3, "Carol", "clip.mp4", video_id="file-123456789abc",
                  view_type="Third-person", source="Simulation",
                  local_path="LabelGUI/Uploads/videos/file-123456789abc.mp4"),
        ])
        # Old dataset, not active, row made before video_id existed (empty).
        self.db.save_dataset("ds0", "Old", "old.xlsx", b"x", "leader", is_active=False)
        self.db.replace_dataset_items("ds0", [
            _item("ds0", 1, "Dan", "https://www.youtube.com/shorts/BBBBBBBBBBB"),
        ])
        self.db.set_active_dataset("ds1")

        self.db.update_dataset_item("ds1:1", status="labeled", labeled_by="alice", locked_by="")

        # A failed session with a stray event: must not be counted.
        self.db.upsert_validation_session(sid="s-failed", youtube_link="https://youtu.be/AAAAAAAAAAA", status="running")
        self.db.insert_validation_event("s-failed", 1, "land", 3.0)
        self.db.finalize_validation_session("s-failed", 0.0, 1, status="failed")

        # The finished session, stored with a different link spelling.
        self.db.upsert_validation_session(sid="s-final", youtube_link="https://youtu.be/AAAAAAAAAAA", status="running")
        for i, ev in enumerate(["takeoff", "minor-crash", "takeoff", "land", "unknown"], 1):
            self.db.insert_validation_event("s-final", i, ev, float(i))
        self.db.finalize_validation_session("s-final", 60.0, 5)

    def tearDown(self):
        self.tmp.cleanup()

    def _rows(self, **kw):
        return {r["video_id"]: r for r in build_video_manifest(self.db_path, **kw)}

    def test_one_row_per_video(self):
        rows = self._rows()
        self.assertEqual(set(rows), {"AAAAAAAAAAA", "file-123456789abc", "BBBBBBBBBBB"})
        self.assertEqual(rows["AAAAAAAAAAA"]["item_count"], 2)

    def test_labeled_row_is_kept_and_events_counted_from_final_session(self):
        r = self._rows()["AAAAAAAAAAA"]
        self.assertEqual(r["status"], "labeled")
        self.assertEqual(r["person_name"], "Alice")
        self.assertEqual(r["labeled_by"], "alice")
        self.assertEqual(r["view_type"], "FPV")
        self.assertEqual(r["source"], "Real")
        self.assertEqual(r["final_sessions"], 1)
        self.assertEqual(r["labeled_sid"], "s-final")
        self.assertEqual((r["n_takeoff"], r["n_land"], r["n_minor_crash"], r["n_severe_crash"], r["n_other"]),
                         (2, 1, 1, 0, 1))
        self.assertEqual(r["events_total"], 5)

    def test_uploaded_file_has_local_path_and_no_link(self):
        r = self._rows()["file-123456789abc"]
        self.assertEqual(r["link"], "")
        self.assertEqual(r["file_name"], "clip.mp4")
        self.assertEqual(r["local_path"], "LabelGUI/Uploads/videos/file-123456789abc.mp4")
        self.assertEqual(r["view_type"], "Third-person")
        self.assertEqual(r["source"], "Simulation")
        self.assertEqual(r["events_total"], 0)

    def test_old_row_without_video_id_gets_youtube_id(self):
        r = self._rows()["BBBBBBBBBBB"]
        self.assertEqual(r["link"], "https://www.youtube.com/shorts/BBBBBBBBBBB")
        self.assertEqual(r["view_type"], "Unknown")
        self.assertEqual(r["status"], "not_labeled")

    def test_active_only(self):
        self.assertEqual(set(self._rows(active_only=True)), {"AAAAAAAAAAA", "file-123456789abc"})

    def test_db_is_not_modified(self):
        before = self.db_path.read_bytes()
        build_video_manifest(self.db_path)
        self.assertEqual(self.db_path.read_bytes(), before)

    def test_read_only_connection(self):
        # Writing through the export's connection mode must be impossible.
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("DELETE FROM dataset_items")
        conn.close()

    def test_csv_has_all_columns(self):
        out = Path(self.tmp.name) / "sub" / "video_manifest.csv"
        write_video_manifest(build_video_manifest(self.db_path), out)
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, COLUMNS)
            self.assertEqual(len(list(reader)), 3)


if __name__ == "__main__":
    unittest.main()
