import os
import sqlite3
from datetime import datetime

def _base_dir():
    return os.path.dirname(os.path.abspath(__file__))

def db_path():
    db_dir = os.path.join(_base_dir(), "db")
    os.makedirs(db_dir, exist_ok=True)
    return os.path.join(db_dir, "droneai.sqlite")

def connect():
    conn = sqlite3.connect(db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = connect()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS videos (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person TEXT NOT NULL,
        scenario TEXT NOT NULL,
        youtube_link TEXT NOT NULL,
        UNIQUE(person, scenario, youtube_link)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS inference_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        sample_fps REAL,
        conf REAL,
        predicted_crash_events INTEGER,
        predicted_crashes_per_min REAL,
        duration_sec REAL,
        video_path TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(video_id) REFERENCES videos(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS verify_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        video_id INTEGER NOT NULL,
        role TEXT NOT NULL,              -- 'label' or 'review'
        user_name TEXT NOT NULL,
        parent_run_id INTEGER,           -- review can point to label run
        status TEXT NOT NULL,            -- 'in_progress', 'saved', 'submitted'
        verified_crash_events INTEGER DEFAULT 0,
        verified_crashes_per_min REAL DEFAULT 0,
        notes TEXT DEFAULT '',
        last_time_sec REAL DEFAULT 0,    -- resume position
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(video_id) REFERENCES videos(id),
        FOREIGN KEY(parent_run_id) REFERENCES verify_runs(id)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS verify_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        t_sec REAL NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES verify_runs(id)
    )
    """)

    conn.commit()
    conn.close()

def now_iso():
    return datetime.utcnow().isoformat()

def get_or_create_video(person, scenario, youtube_link):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO videos(person, scenario, youtube_link)
        VALUES(?,?,?)
    """, (person, scenario, youtube_link))
    conn.commit()
    cur.execute("""
        SELECT id FROM videos WHERE person=? AND scenario=? AND youtube_link=?
    """, (person, scenario, youtube_link))
    row = cur.fetchone()
    conn.close()
    return int(row["id"])

def insert_inference(video_id, sample_fps, conf, pred_events, pred_per_min, duration_sec, video_path):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO inference_runs(video_id, sample_fps, conf, predicted_crash_events, predicted_crashes_per_min,
                                   duration_sec, video_path, created_at)
        VALUES(?,?,?,?,?,?,?,?)
    """, (video_id, sample_fps, conf, pred_events, pred_per_min, duration_sec, video_path, now_iso()))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)

def latest_inference(video_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM inference_runs WHERE video_id=? ORDER BY id DESC LIMIT 1
    """, (video_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def create_verify_run(video_id, role, user_name, parent_run_id=None):
    conn = connect()
    cur = conn.cursor()
    t = now_iso()
    cur.execute("""
        INSERT INTO verify_runs(video_id, role, user_name, parent_run_id, status, created_at, updated_at)
        VALUES(?,?,?,?,?,?,?)
    """, (video_id, role, user_name, parent_run_id, "in_progress", t, t))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return int(rid)

def set_verify_status(run_id, status, last_time_sec=None, notes=None):
    conn = connect()
    cur = conn.cursor()
    fields = ["status=?", "updated_at=?"]
    vals = [status, now_iso()]
    if last_time_sec is not None:
        fields.append("last_time_sec=?")
        vals.append(float(last_time_sec))
    if notes is not None:
        fields.append("notes=?")
        vals.append(notes)
    vals.append(run_id)
    cur.execute(f"UPDATE verify_runs SET {', '.join(fields)} WHERE id=?", vals)
    conn.commit()
    conn.close()

def add_verify_event(run_id, t_sec):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO verify_events(run_id, t_sec, created_at) VALUES(?,?,?)
    """, (run_id, float(t_sec), now_iso()))
    conn.commit()
    conn.close()

def compute_verify_counts(run_id, duration_sec):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS n FROM verify_events WHERE run_id=?", (run_id,))
    n = int(cur.fetchone()["n"])
    per_min = (n / (duration_sec / 60.0)) if duration_sec and duration_sec > 0 else 0.0
    cur.execute("""
        UPDATE verify_runs SET verified_crash_events=?, verified_crashes_per_min=?, updated_at=?
        WHERE id=?
    """, (n, float(per_min), now_iso(), run_id))
    conn.commit()
    conn.close()
    return n, per_min

def get_verify_run(run_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("SELECT * FROM verify_runs WHERE id=?", (run_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None

def list_verify_runs_for_video(video_id):
    conn = connect()
    cur = conn.cursor()
    cur.execute("""
        SELECT * FROM verify_runs WHERE video_id=? ORDER BY id DESC
    """, (video_id,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

