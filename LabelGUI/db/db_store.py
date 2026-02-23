# db/db_store.py
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


class DBStore:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path))
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self):
        with self._conn() as conn:
            # -------------------------
            # Crash verification (existing)
            # -------------------------
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

            # -------------------------
            # Validation (new)
            # -------------------------
            conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_sessions (
                sid TEXT PRIMARY KEY,
                person_name TEXT DEFAULT '',
                scenario_base TEXT DEFAULT '',
                youtube_link TEXT NOT NULL,
                video_path TEXT DEFAULT '',
                folder_path TEXT DEFAULT '',
                delete_original INTEGER DEFAULT 0,
                duration_sec REAL DEFAULT 0.0,
                events_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running',  -- running | final | failed
                created_at TEXT,
                updated_at TEXT
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS validation_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sid TEXT NOT NULL,
                idx INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                time_sec REAL NOT NULL,
                created_at TEXT,
                FOREIGN KEY (sid) REFERENCES validation_sessions(sid)
            )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_validation_events_sid ON validation_events(sid);")

            # -------------------------
            # Training (new)
            # -------------------------
            conn.execute("""
            CREATE TABLE IF NOT EXISTS training_sessions (
                sid TEXT PRIMARY KEY,
                parent_sid TEXT DEFAULT '',
                user_name TEXT DEFAULT '',
                youtube_link TEXT NOT NULL,
                video_path TEXT DEFAULT '',
                metadata_path TEXT DEFAULT '',
                capture_mode TEXT DEFAULT '',
                custom_fps REAL DEFAULT 0.0,
                delete_original INTEGER DEFAULT 0,
                keep_metadata INTEGER DEFAULT 0,
                status TEXT DEFAULT 'running', -- running | paused | final | failed
                created_at TEXT,
                updated_at TEXT
            )
            """)

            # store chunks instead of per-frame events (fast + small)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS training_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sid TEXT NOT NULL,
                start_frame INTEGER NOT NULL,
                end_frame INTEGER NOT NULL,
                label TEXT NOT NULL,
                created_at TEXT,
                FOREIGN KEY (sid) REFERENCES training_sessions(sid)
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_chunks_sid ON training_chunks(sid);")

            conn.commit()

    # =========================
    # Crash verification API (kept for compatibility)
    # =========================
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

    def get_session(self, sid: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("SELECT * FROM verifications WHERE sid=?", (sid,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_latest_by_key(self, person: str, scenario: str, youtube_link: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT * FROM verifications
                WHERE person=? AND scenario=? AND youtube_link=?
                ORDER BY updated_at DESC
                LIMIT 1
            """, (person, scenario, youtube_link))
            row = cur.fetchone()
            return dict(row) if row else None

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

    def list_all(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT person, scenario, youtube_link,
                       pred_crash_events, pred_crashes_per_min,
                       verified_crash_events, verified_crashes_per_min,
                       status, notes, updated_at
                FROM verifications
                ORDER BY updated_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]

    # =========================
    # Validation API (new)
    # =========================
    def upsert_validation_session(self, **row):
        now = datetime.utcnow().isoformat()
        if "created_at" not in row:
            row["created_at"] = now
        row["updated_at"] = now

        cols = list(row.keys())
        vals = [row[c] for c in cols]
        q_cols = ", ".join(cols)
        q_marks = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "sid"])
        sql = f"""
        INSERT INTO validation_sessions ({q_cols})
        VALUES ({q_marks})
        ON CONFLICT(sid) DO UPDATE SET {updates}
        """
        with self._conn() as conn:
            conn.execute(sql, vals)
            conn.commit()

    def insert_validation_event(self, sid: str, idx: int, event_type: str, time_sec: float):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO validation_events (sid, idx, event_type, time_sec, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (sid, idx, event_type, float(time_sec), datetime.utcnow().isoformat()))
            conn.commit()

    def finalize_validation_session(self, sid: str, duration_sec: float, events_count: int, status: str = "final"):
        with self._conn() as conn:
            conn.execute("""
                UPDATE validation_sessions
                SET duration_sec=?, events_count=?, status=?, updated_at=?
                WHERE sid=?
            """, (float(duration_sec), int(events_count), status, datetime.utcnow().isoformat(), sid))
            conn.commit()

    # =========================
    # Training API (new)
    # =========================
    def upsert_training_session(self, **row):
        now = datetime.utcnow().isoformat()
        if "created_at" not in row:
            row["created_at"] = now
        row["updated_at"] = now

        cols = list(row.keys())
        vals = [row[c] for c in cols]
        q_cols = ", ".join(cols)
        q_marks = ", ".join(["?"] * len(cols))
        updates = ", ".join([f"{c}=excluded.{c}" for c in cols if c != "sid"])
        sql = f"""
        INSERT INTO training_sessions ({q_cols})
        VALUES ({q_marks})
        ON CONFLICT(sid) DO UPDATE SET {updates}
        """
        with self._conn() as conn:
            conn.execute(sql, vals)
            conn.commit()

    def replace_training_chunks(self, sid: str, chunks: List[Dict[str, Any]]):
        """
        chunks: [{"start_frame": int, "end_frame": int, "label": str}, ...]
        """
        with self._conn() as conn:
            conn.execute("DELETE FROM training_chunks WHERE sid=?", (sid,))
            now = datetime.utcnow().isoformat()
            for c in chunks:
                conn.execute("""
                    INSERT INTO training_chunks (sid, start_frame, end_frame, label, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (sid, int(c["start_frame"]), int(c["end_frame"]), str(c["label"]), now))
            conn.commit()
