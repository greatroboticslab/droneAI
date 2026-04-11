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
            # Existing tables
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
                status TEXT DEFAULT 'running',
                notes TEXT DEFAULT '',
                resume_offset_sec REAL DEFAULT 0.0,
                sample_fps REAL DEFAULT 2.0,
                conf REAL DEFAULT 0.5,
                updated_at TEXT
            )
            """)

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
                status TEXT DEFAULT 'running',
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
                status TEXT DEFAULT 'running',
                created_at TEXT,
                updated_at TEXT
            )
            """)

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

            # -------------------------
            # NEW: shared datasets
            # -------------------------
            conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                dataset_key TEXT PRIMARY KEY,
                dataset_name TEXT NOT NULL,
                original_filename TEXT NOT NULL,
                file_blob BLOB NOT NULL,
                uploaded_by TEXT DEFAULT '',
                uploaded_at TEXT,
                is_active INTEGER DEFAULT 0
            )
            """)

            conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_items (
                item_key TEXT PRIMARY KEY,
                dataset_key TEXT NOT NULL,
                row_index INTEGER NOT NULL,
                person_name TEXT DEFAULT '',
                youtube_link TEXT NOT NULL,
                status TEXT DEFAULT 'not_labeled',
                labeled_by TEXT DEFAULT '',
                locked_by TEXT DEFAULT '',
                scenario_type TEXT DEFAULT '',
                updated_at TEXT,
                UNIQUE(dataset_key, row_index),
                FOREIGN KEY (dataset_key) REFERENCES datasets(dataset_key)
            )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_items_dataset_key ON dataset_items(dataset_key);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dataset_items_status ON dataset_items(status);")

            conn.commit()

    # -------------------------
    # Existing APIs
    # -------------------------
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
        with self._conn() as conn:
            conn.execute("DELETE FROM training_chunks WHERE sid=?", (sid,))
            now = datetime.utcnow().isoformat()
            for c in chunks:
                conn.execute("""
                    INSERT INTO training_chunks (sid, start_frame, end_frame, label, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (sid, int(c["start_frame"]), int(c["end_frame"]), str(c["label"]), now))
            conn.commit()

    # -------------------------
    # NEW: dataset library
    # -------------------------
    def save_dataset(self, dataset_key: str, dataset_name: str, original_filename: str,
                     file_blob: bytes, uploaded_by: str, is_active: bool = False):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            if is_active:
                conn.execute("UPDATE datasets SET is_active=0")
            conn.execute("""
                INSERT INTO datasets (
                    dataset_key, dataset_name, original_filename, file_blob,
                    uploaded_by, uploaded_at, is_active
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(dataset_key) DO UPDATE SET
                    dataset_name=excluded.dataset_name,
                    original_filename=excluded.original_filename,
                    file_blob=excluded.file_blob,
                    uploaded_by=excluded.uploaded_by,
                    uploaded_at=excluded.uploaded_at,
                    is_active=excluded.is_active
            """, (
                dataset_key, dataset_name, original_filename,
                file_blob, uploaded_by, now, 1 if is_active else 0
            ))
            conn.commit()

    def dataset_exists(self, dataset_key: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("SELECT 1 FROM datasets WHERE dataset_key=?", (dataset_key,))
            return cur.fetchone() is not None

    def list_datasets(self) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT dataset_key, dataset_name, original_filename, uploaded_by, uploaded_at, is_active
                FROM datasets
                ORDER BY uploaded_at DESC
            """)
            return [dict(r) for r in cur.fetchall()]

    def get_dataset(self, dataset_key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT dataset_key, dataset_name, original_filename, uploaded_by, uploaded_at, is_active
                FROM datasets WHERE dataset_key=?
            """, (dataset_key,))
            row = cur.fetchone()
            return dict(row) if row else None

    def get_active_dataset(self) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT dataset_key, dataset_name, original_filename, uploaded_by, uploaded_at, is_active
                FROM datasets WHERE is_active=1
                ORDER BY uploaded_at DESC LIMIT 1
            """)
            row = cur.fetchone()
            return dict(row) if row else None

    def set_active_dataset(self, dataset_key: str):
        with self._conn() as conn:
            conn.execute("UPDATE datasets SET is_active=0")
            conn.execute("UPDATE datasets SET is_active=1 WHERE dataset_key=?", (dataset_key,))
            conn.commit()

    def get_dataset_file_blob(self, dataset_key: str) -> Optional[bytes]:
        with self._conn() as conn:
            cur = conn.execute("SELECT file_blob FROM datasets WHERE dataset_key=?", (dataset_key,))
            row = cur.fetchone()
            return row["file_blob"] if row else None

    def replace_dataset_items(self, dataset_key: str, items: List[Dict[str, Any]]):
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("DELETE FROM dataset_items WHERE dataset_key=?", (dataset_key,))
            for item in items:
                conn.execute("""
                    INSERT INTO dataset_items (
                        item_key, dataset_key, row_index, person_name, youtube_link,
                        status, labeled_by, locked_by, scenario_type, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    item["item_key"],
                    dataset_key,
                    int(item["row_index"]),
                    item.get("person_name", ""),
                    item.get("youtube_link", ""),
                    item.get("status", "not_labeled"),
                    item.get("labeled_by", ""),
                    item.get("locked_by", ""),
                    item.get("scenario_type", ""),
                    now,
                ))
            conn.commit()

    def list_dataset_items(self, dataset_key: str) -> List[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT item_key, dataset_key, row_index, person_name, youtube_link,
                       status, labeled_by, locked_by, scenario_type, updated_at
                FROM dataset_items
                WHERE dataset_key=?
                ORDER BY row_index ASC
            """, (dataset_key,))
            return [dict(r) for r in cur.fetchall()]

    def get_dataset_item(self, item_key: str) -> Optional[Dict[str, Any]]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT item_key, dataset_key, row_index, person_name, youtube_link,
                       status, labeled_by, locked_by, scenario_type, updated_at
                FROM dataset_items
                WHERE item_key=?
            """, (item_key,))
            row = cur.fetchone()
            return dict(row) if row else None

    def update_dataset_item(self, item_key: str, status: Optional[str] = None,
                            labeled_by: Optional[str] = None,
                            locked_by: Optional[str] = None,
                            scenario_type: Optional[str] = None):
        row = self.get_dataset_item(item_key)
        if not row:
            return
        new_status = row["status"] if status is None else status
        new_labeled_by = row["labeled_by"] if labeled_by is None else labeled_by
        new_locked_by = row["locked_by"] if locked_by is None else locked_by
        new_scenario_type = row["scenario_type"] if scenario_type is None else scenario_type
        with self._conn() as conn:
            conn.execute("""
                UPDATE dataset_items
                SET status=?, labeled_by=?, locked_by=?, scenario_type=?, updated_at=?
                WHERE item_key=?
            """, (
                new_status,
                new_labeled_by,
                new_locked_by,
                new_scenario_type,
                datetime.utcnow().isoformat(),
                item_key,
            ))
            conn.commit()

    def dataset_stats(self, dataset_key: str) -> Dict[str, int]:
        with self._conn() as conn:
            cur = conn.execute("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='not_labeled' THEN 1 ELSE 0 END) AS not_labeled,
                    SUM(CASE WHEN status='in_progress' THEN 1 ELSE 0 END) AS in_progress,
                    SUM(CASE WHEN status='labeled' THEN 1 ELSE 0 END) AS labeled
                FROM dataset_items
                WHERE dataset_key=?
            """, (dataset_key,))
            row = cur.fetchone()
            return {
                "total": int(row["total"] or 0),
                "not_labeled": int(row["not_labeled"] or 0),
                "in_progress": int(row["in_progress"] or 0),
                "labeled": int(row["labeled"] or 0),
            }
