# LabelGUI/db/db_store.py
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

def _repo_root():
    # LabelGUI/db/db_store.py -> repo root is 2 levels up
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def db_path():
    # keep DB inside LabelGUI/db/
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, "droneai.sqlite")

@contextmanager
def connect():
    path = db_path()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def utcnow():
    return datetime.utcnow().isoformat()

def init_db():
    with connect() as conn:
        cur = conn.cursor()

        # videos: one per (person, scenario, youtube_link)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person TEXT NOT NULL,
            scenario TEXT NOT NULL,
            youtube_link TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(person, scenario, youtube_link)
        )
        """)

        # inference results (latest result kept, can be extended later)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS inference_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            crash_events INTEGER,
            crashes_per_min REAL,
            duration_sec REAL,
            video_path TEXT,
            model_weights TEXT,
            sample_fps REAL,
            conf REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
        """)

        # labeling/review runs
        cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            video_id INTEGER NOT NULL,
            run_type TEXT NOT NULL,     -- "label" or "review"
            labeler TEXT,               -- who did it
            status TEXT NOT NULL,       -- "in_progress" | "paused" | "submitted"
            verified_crash_events INTEGER NOT NULL DEFAULT 0,
            notes TEXT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            last_time_sec REAL NOT NULL DEFAULT 0,
            FOREIGN KEY(video_id) REFERENCES videos(id)
        )
        """)

        # crash click timestamps for each run
        cur.execute("""
        CREATE TABLE IF NOT EXISTS verification_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            t_sec REAL NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(run_id) REFERENCES verification_runs(id)
        )
        """)

def get_or_create_video(person: str, scenario: str, youtube_link: str) -> int:
    with connect() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT id FROM videos WHERE person=? AND scenario=? AND youtube_link=?",
            (person, scenario, youtube_link),
        )
        row = cur.fetchone()
        if row:
            return int(row["id"])

        cur.execute(
            "INSERT INTO videos(person, scenario, youtube_link, created_at) VALUES(?,?,?,?)",
            (person, scenario, youtube_link, utcnow()),
        )
        return int(cur.lastrowid)

def add_inference_result(video_id: int, crash_events, crashes_per_min, duration_sec, video_path,
                         model_weights, sample_fps, conf):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO inference_results(
              video_id, crash_events, crashes_per_min, duration_sec, video_path,
              model_weights, sample_fps, conf, created_at
            ) VALUES(?,?,?,?,?,?,?,?,?)
        """, (
            video_id,
            None if crash_events == "" else int(crash_events),
            None if crashes_per_min == "" else float(crashes_per_min),
            None if duration_sec == "" else float(duration_sec),
            video_path,
            model_weights,
            float(sample_fps) if sample_fps is not None else None,
            float(conf) if conf is not None else None,
            utcnow(),
        ))

def latest_inference(video_id: int):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT * FROM inference_results
            WHERE video_id=?
            ORDER BY id DESC
            LIMIT 1
        """, (video_id,))
        r = cur.fetchone()
        return dict(r) if r else None

def create_run(video_id: int, run_type="label", labeler=""):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verification_runs(video_id, run_type, labeler, status, started_at)
            VALUES(?,?,?,?,?)
        """, (video_id, run_type, labeler, "in_progress", utcnow()))
        return int(cur.lastrowid)

def load_run(run_id: int):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM verification_runs WHERE id=?", (run_id,))
        r = cur.fetchone()
        return dict(r) if r else None

def load_video(video_id: int):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM videos WHERE id=?", (video_id,))
        r = cur.fetchone()
        return dict(r) if r else None

def set_run_status(run_id: int, status: str, last_time_sec=None, notes=None):
    with connect() as conn:
        cur = conn.cursor()
        if last_time_sec is None and notes is None:
            cur.execute("UPDATE verification_runs SET status=? WHERE id=?", (status, run_id))
        elif last_time_sec is None:
            cur.execute("UPDATE verification_runs SET status=?, notes=? WHERE id=?", (status, notes, run_id))
        elif notes is None:
            cur.execute("UPDATE verification_runs SET status=?, last_time_sec=? WHERE id=?", (status, float(last_time_sec), run_id))
        else:
            cur.execute("UPDATE verification_runs SET status=?, last_time_sec=?, notes=? WHERE id=?",
                        (status, float(last_time_sec), notes, run_id))

def finish_run(run_id: int, notes: str):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            UPDATE verification_runs
            SET status=?, notes=?, finished_at=?
            WHERE id=?
        """, ("submitted", notes, utcnow(), run_id))

def increment_run_count(run_id: int, t_sec: float):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO verification_events(run_id, t_sec, created_at)
            VALUES(?,?,?)
        """, (run_id, float(t_sec), utcnow()))
        cur.execute("""
            UPDATE verification_runs
            SET verified_crash_events = verified_crash_events + 1
            WHERE id=?
        """, (run_id,))

def list_events(run_id: int):
    with connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            SELECT t_sec FROM verification_events
            WHERE run_id=?
            ORDER BY id ASC
        """, (run_id,))
        return [float(r["t_sec"]) for r in cur.fetchall()]
