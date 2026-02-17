# db/db_store.py
import sqlite3
from pathlib import Path
from datetime import datetime

class DBStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init(self):
        with self._conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS verifications (
                sid TEXT PRIMARY KEY,
                person TEXT NOT NULL,
                scenario TEXT NOT NULL,
                youtube_link TEXT NOT NULL,
                video_path TEXT DEFAULT '',
                pred_crash_events INTEGER DEFAULT 0,
                pred_crashes_per_min REAL DEFAULT 0.0,
                duration_sec REAL DEFAULT 0.0,
                verified_crash_events INTEGER DEFAULT 0,
                verified_crashes_per_min REAL DEFAULT 0.0,
                status TEXT DEFAULT 'running',  -- running | saved | final
                notes TEXT DEFAULT '',
                resume_offset_sec REAL DEFAULT 0.0,
                sample_fps REAL DEFAULT 2.0,
                conf REAL DEFAULT 0.5,
                updated_at TEXT
            )
            """)
            conn.commit()

    def upsert_session(self, **row):
        row["updated_at"] = datetime.utcnow().isoformat()
        cols = list(row.keys())
        vals = [row[c] for c in cols]
        q_cols = ", ".join(cols)
        q_marks = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "sid"])
        sql = f"""
        INSERT INTO verifications ({q_cols})
        VALUES ({q_marks})
        ON CONFLICT(sid) DO UPDATE SET {updates}
        """
        with self._conn() as conn:
            conn.execute(sql, vals)
            conn.commit()

    def get_session(self, sid: str):
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM verifications WHERE sid=?", (sid,))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def get_latest_by_key(self, person: str, scenario: str, youtube_link: str):
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT * FROM verifications
                WHERE person=? AND scenario=? AND youtube_link=?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (person, scenario, youtube_link))
            row = cur.fetchone()
            if not row:
                return None
            cols = [d[0] for d in cur.description]
            return dict(zip(cols, row))

    def update_counts(self, sid: str, verified_crash_events: int, verified_crashes_per_min: float):
        with self._conn() as conn:
            conn.execute("""
                UPDATE verifications
                SET verified_crash_events=?, verified_crashes_per_min=?, updated_at=?
                WHERE sid=?
            """, (verified_crash_events, verified_crashes_per_min, datetime.utcnow().isoformat(), sid))
            conn.commit()

    def update_status_notes(self, sid: str, status: str, notes: str, resume_offset_sec: float):
        with self._conn() as conn:
            conn.execute("""
                UPDATE verifications
                SET status=?, notes=?, resume_offset_sec=?, updated_at=?
                WHERE sid=?
            """, (status, notes, resume_offset_sec, datetime.utcnow().isoformat(), sid))
            conn.commit()

    def list_all(self):
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT person, scenario, youtube_link,
                       pred_crash_events, pred_crashes_per_min,
                       verified_crash_events, verified_crashes_per_min,
                       status, notes, updated_at
                FROM verifications
                ORDER BY updated_at DESC
            """)
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, r)) for r in cur.fetchall()]
