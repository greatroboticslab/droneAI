"""
Run from the repo root:  python -m unittest discover -s LabelGUI/tests
Reviewer name on every event, and watched_until_sec on sessions (DB level).
"""
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from clip_dataset import load_labeled_sessions  # noqa: E402
from db.db_store import DBStore  # noqa: E402


class EventReviewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "t.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_old_db_gets_new_columns(self):
        # A DB from before these columns existed.
        conn = sqlite3.connect(self.path)
        conn.execute("CREATE TABLE validation_sessions (sid TEXT PRIMARY KEY, youtube_link TEXT NOT NULL, "
                     "status TEXT DEFAULT 'running', duration_sec REAL DEFAULT 0, events_count INTEGER DEFAULT 0, "
                     "updated_at TEXT)")
        conn.execute("CREATE TABLE validation_events (id INTEGER PRIMARY KEY AUTOINCREMENT, sid TEXT NOT NULL, "
                     "idx INTEGER NOT NULL, event_type TEXT NOT NULL, time_sec REAL NOT NULL, created_at TEXT)")
        conn.execute("INSERT INTO validation_events (sid, idx, event_type, time_sec) VALUES ('old', 1, 'land', 2.0)")
        conn.commit()
        conn.close()

        DBStore(str(self.path))
        conn = sqlite3.connect(self.path)
        ev_cols = {r[1] for r in conn.execute("PRAGMA table_info(validation_events)")}
        se_cols = {r[1] for r in conn.execute("PRAGMA table_info(validation_sessions)")}
        old = conn.execute("SELECT reviewer FROM validation_events WHERE sid='old'").fetchone()
        conn.close()
        self.assertIn("reviewer", ev_cols)
        self.assertIn("watched_until_sec", se_cols)
        self.assertEqual(old[0], "")  # old marks keep an empty reviewer

    def test_reviewer_and_watched_time_are_stored(self):
        db = DBStore(str(self.path))
        db.save_dataset("ds", "D", "d.xlsx", b"x", "leader", is_active=True)
        db.upsert_validation_session(sid="s1", youtube_link="https://youtu.be/AAAAAAAAAAA",
                                     folder_path="LabelGUI/ValidationResults/P", status="running")
        db.insert_validation_event("s1", 1, "takeoff", 1.5, reviewer="alice")
        db.insert_validation_event("s1", 2, "land", 20.0)  # no reviewer given
        db.finalize_validation_session("s1", 120.0, 2, watched_until_sec=45.0)

        conn = sqlite3.connect(self.path)
        rows = conn.execute("SELECT idx, reviewer FROM validation_events ORDER BY idx").fetchall()
        watched, duration = conn.execute("SELECT watched_until_sec, duration_sec FROM validation_sessions").fetchone()
        conn.close()
        self.assertEqual(rows, [(1, "alice"), (2, "")])
        self.assertEqual((watched, duration), (45.0, 120.0))

        s = load_labeled_sessions(self.path)[0]
        self.assertEqual(s["watched_until_sec"], 45.0)
        self.assertEqual(s["reviewers"], {1: "alice", 2: ""})

    def test_finalize_without_watched_time_keeps_old_behaviour(self):
        db = DBStore(str(self.path))
        db.upsert_validation_session(sid="s1", youtube_link="x", status="running")
        db.finalize_validation_session("s1", 30.0, 0, status="cancelled")
        conn = sqlite3.connect(self.path)
        row = conn.execute("SELECT status, watched_until_sec FROM validation_sessions").fetchone()
        conn.close()
        self.assertEqual(row, ("cancelled", 0.0))


if __name__ == "__main__":
    unittest.main()
