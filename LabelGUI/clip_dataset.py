"""
Re-cut event clips from the labels stored in the DB, without relabeling.

Each finished labeling session has its event marks (validation_events: type and
time). This script cuts a new clip around every mark with a window you choose,
straight from the source video, so the window can change later without anyone
watching the videos again.

    python LabelGUI/clip_dataset.py --dry-run                 # list what would be cut
    python LabelGUI/clip_dataset.py                           # -2 s / +10 s (default)
    python LabelGUI/clip_dataset.py --before 3 --after 8 --run-name w3_8
    python LabelGUI/clip_dataset.py --download                # fetch videos that are gone

Output (default LabelGUI/RecutClips/<run-name>/), same layout as ValidationResults:
    <session_name>/clips/<idx>_<event>.mp4
    clips.csv            one row per clip: video_id, view_type, source, event, times, path
    recut_summary.json   window, counts, and sessions that were skipped (with the reason)

Only the newest finished session per video is used, unless --all-sessions.
The DB is opened read-only.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

from repo_paths import LABELGUI_DIR, repo_rel, resolve_path
from video_manifest import (
    DEFAULT_DB,
    _connect_read_only,
    _load_items,
    item_video_id,
    link_key,
    build_video_manifest,
    session_folder_name,
)

EVENT_CLASSES = ("takeoff", "land", "minor-crash", "severe-crash")
DOWNLOAD_DIR = LABELGUI_DIR / "YouTubeDownloads"
UPLOAD_DIR = LABELGUI_DIR / "Uploads" / "videos"
DEFAULT_OUT_ROOT = LABELGUI_DIR / "RecutClips"

CLIP_COLUMNS = [
    "clip_path", "video_id", "view_type", "source", "session_name", "sid", "event_idx",
    "event_type", "event_time_sec", "reviewer", "clip_start_sec", "clip_end_sec", "before_sec",
    "after_sec", "fps", "frames", "video_path",
]


# ---------------------------------------------------------------- sessions from the DB
def load_labeled_sessions(db_path, all_sessions=False):
    """
    Finished sessions with their events, one dict per session:
    sid, session_name, folder_path, youtube_link, video_path, duration_sec,
    watched_until_sec (0 = not recorded), updated_at, video_id, view_type, source,
    local_path, events=[(idx, event_type, time_sec), ...], reviewers={idx: name}

    Newest finished session per video only, unless all_sessions=True.
    """
    manifest = {r["video_id"]: r for r in build_video_manifest(db_path)}
    conn = _connect_read_only(db_path)
    try:
        items = _load_items(conn, active_only=False)
        vid_by_key = {}
        local_by_vid = {}
        for it in items:
            vid = item_video_id(it)
            vid_by_key.setdefault(link_key(it["youtube_link"]), vid)
            if it["local_path"]:
                local_by_vid.setdefault(vid, it["local_path"])

        sessions = []
        # SELECT * so DBs made before watched_until_sec / reviewer existed still work.
        for r in conn.execute(
            "SELECT * FROM validation_sessions WHERE status = 'final' "
            "ORDER BY COALESCE(updated_at, created_at, '') DESC"
        ):
            r = dict(r)
            link = (r["youtube_link"] or "").strip()
            vid = vid_by_key.get(link_key(link)) or item_video_id({"youtube_link": link})
            video = manifest.get(vid, {})
            event_rows = [dict(e) for e in conn.execute(
                "SELECT * FROM validation_events WHERE sid = ? ORDER BY time_sec", (r["sid"],)
            )]
            events = [(e["idx"], e["event_type"], float(e["time_sec"])) for e in event_rows]
            reviewers = {e["idx"]: e.get("reviewer") or "" for e in event_rows}
            sessions.append({
                "sid": r["sid"],
                "session_name": session_folder_name(r["folder_path"]) if r["folder_path"] else r["sid"][:8],
                "folder_path": (r["folder_path"] or "").replace("\\", "/"),
                "youtube_link": link,
                "video_path": (r["video_path"] or "").replace("\\", "/"),
                "duration_sec": float(r["duration_sec"] or 0.0),
                "watched_until_sec": float(r.get("watched_until_sec") or 0.0),
                "updated_at": r["updated_at"] or "",
                "video_id": vid,
                "view_type": video.get("view_type", "Unknown"),
                "source": video.get("source", "Unknown"),
                "local_path": local_by_vid.get(vid, ""),
                "events": events,
                "reviewers": reviewers,
            })
    finally:
        conn.close()

    if all_sessions:
        return sessions
    newest = {}
    for s in sessions:  # already newest first
        newest.setdefault(s["video_id"], s)
    return list(newest.values())


# ---------------------------------------------------------------- finding the video file
def find_video_file(session, download_dir=DOWNLOAD_DIR):
    """Path of the session's source video on this machine, or None."""
    for rel in (session.get("local_path"), session.get("video_path")):
        if rel:
            p = resolve_path(rel)
            if p is not None and p.exists():
                return p
    vid = session.get("video_id", "")
    if vid and not vid.startswith(("url-", "file-")):
        download_dir = Path(download_dir)
        if download_dir.exists():
            # yt-dlp names downloads "<title>-<id>.<ext>"
            for p in sorted(download_dir.glob(f"*{vid}*")):
                if p.suffix.lower() in (".mp4", ".mkv", ".webm", ".mov") and not p.name.endswith(".part"):
                    return p
    return None


def download_video(link, download_dir=DOWNLOAD_DIR):
    """Download a YouTube link with yt-dlp. Returns the file path or raises."""
    import yt_dlp

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/bv*+ba/b",
        "merge_output_format": "mp4",
        "outtmpl": str(download_dir / "%(title)s-%(id)s.%(ext)s"),
        "quiet": True,
        "noprogress": True,
        "no_warnings": True,
        "noplaylist": True,
    }
    try:
        import imageio_ffmpeg
        opts["ffmpeg_location"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(link, download=True)
        path = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
    if not path.exists():
        matches = sorted(download_dir.glob(f"*{info.get('id', '')}*.mp4"))
        if not matches:
            raise FileNotFoundError(f"yt-dlp finished but no file found for {link}")
        path = matches[0]
    return path


# ---------------------------------------------------------------- cutting
def clip_window(event_time, before, after, duration):
    """(start, end) seconds, clamped to the video."""
    start = max(0.0, event_time - before)
    end = event_time + after
    if duration and duration > 0:
        end = min(end, duration)
    return start, max(start, end)


def cut_segments(cap, fps, windows):
    """
    Write several frame windows of one video in a single front-to-back pass.

    windows: list of (start_sec, end_sec, out_path). Frames are decoded in order
    and never seeked, because OpenCV seeking can land a frame or more off (worse
    on YouTube videos with few keyframes). Overlapping windows share the decode.
    Returns the number of frames written per window, in the same order.
    """
    import cv2

    spans = [(int(round(s * fps)), int(round(e * fps)), Path(p)) for s, e, p in windows]
    written = [0] * len(spans)
    writers = [None] * len(spans)
    last = max((e for _, e, _ in spans), default=-1)
    frame_idx = 0
    try:
        while frame_idx <= last:
            ok, frame = cap.read()
            if not ok:
                break
            for k, (s, e, path) in enumerate(spans):
                if s <= frame_idx <= e:
                    if writers[k] is None:
                        h, w = frame.shape[:2]
                        path.parent.mkdir(parents=True, exist_ok=True)
                        writers[k] = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                    writers[k].write(frame)
                    written[k] += 1
                elif frame_idx > e and writers[k] is not None:
                    writers[k].release()
                    writers[k] = None
            frame_idx += 1
    finally:
        for wtr in writers:
            if wtr is not None:
                wtr.release()
    return written


def open_video(path):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0.0, 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frames / fps if fps else 0.0
    return cap, float(fps), float(duration)


def recut_session(session, video_path, out_dir, before, after, classes=EVENT_CLASSES, dry_run=False):
    """Cut one clip per event of a session. Returns (clip rows, skipped event count)."""
    rows, skipped = [], 0
    events = [e for e in session["events"] if e[1] in classes]
    skipped = len(session["events"]) - len(events)
    if not events:
        return rows, skipped

    cap, fps, duration = (None, 0.0, session["duration_sec"]) if dry_run else open_video(video_path)
    if not dry_run and cap is None:
        raise RuntimeError(f"cannot open video {video_path}")
    duration = duration or session["duration_sec"]
    try:
        planned = []
        for idx, event_type, t in events:
            start, end = clip_window(t, before, after, duration)
            rel_clip = Path(session["session_name"]) / "clips" / f"{int(idx):03d}_{event_type}.mp4"
            planned.append((idx, event_type, t, start, end, rel_clip))
        frame_counts = [0] * len(planned)
        if not dry_run:
            frame_counts = cut_segments(cap, fps, [(p[3], p[4], out_dir / p[5]) for p in planned])

        for (idx, event_type, t, start, end, rel_clip), frames in zip(planned, frame_counts):
            if not dry_run and frames == 0:
                skipped += 1
                continue
            rows.append({
                "clip_path": repo_rel(out_dir / rel_clip),
                "video_id": session["video_id"],
                "view_type": session["view_type"],
                "source": session["source"],
                "session_name": session["session_name"],
                "sid": session["sid"],
                "event_idx": int(idx),
                "event_type": event_type,
                "event_time_sec": round(t, 3),
                "reviewer": session.get("reviewers", {}).get(idx, ""),
                "clip_start_sec": round(start, 3),
                "clip_end_sec": round(end, 3),
                "before_sec": before,
                "after_sec": after,
                "fps": round(fps, 3),
                "frames": frames,
                "video_path": repo_rel(video_path) if video_path else "",
            })
    finally:
        if cap is not None:
            cap.release()
    return rows, skipped


def write_rows_csv(rows, path, columns):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def prepare_sessions(args):
    """Shared CLI step: load sessions, then find (or download) each video."""
    db_path = resolve_path(args.db, must_exist=True)
    sessions = load_labeled_sessions(db_path, all_sessions=args.all_sessions)
    if args.limit:
        sessions = sessions[: args.limit]

    ready, missing = [], []
    for s in sessions:
        path = find_video_file(s)
        if path is None and args.download and s["youtube_link"].startswith("http"):
            try:
                print(f"Downloading {s['youtube_link']} ...")
                path = download_video(s["youtube_link"])
            except Exception as e:  # keep going; report at the end
                missing.append({"sid": s["sid"], "video_id": s["video_id"], "reason": f"download failed: {e}"})
                continue
        if path is None and not args.dry_run:
            missing.append({"sid": s["sid"], "video_id": s["video_id"],
                            "reason": "video file not found (use --download for YouTube links)"})
            continue
        ready.append((s, path))
    return db_path, ready, missing


def add_common_args(parser, default_run):
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB (default: db/droneai.sqlite)")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Parent folder for runs")
    parser.add_argument("--run-name", default=default_run)
    parser.add_argument("--all-sessions", action="store_true",
                        help="Use every finished session, not only the newest per video")
    parser.add_argument("--download", action="store_true", help="Download YouTube videos that are not on disk")
    parser.add_argument("--limit", type=int, default=0, help="Only the first N sessions (for quick tests)")
    parser.add_argument("--dry-run", action="store_true", help="List what would be done, write no video")


def main():
    parser = argparse.ArgumentParser(description="Re-cut event clips from stored labels.")
    parser.add_argument("--before", type=float, default=2.0, help="Seconds before the event (default 2)")
    parser.add_argument("--after", type=float, default=10.0, help="Seconds after the event (default 10)")
    add_common_args(parser, default_run="")
    args = parser.parse_args()
    if args.before < 0 or args.after <= 0:
        parser.error("--before must be >= 0 and --after > 0")

    run_name = args.run_name or f"recut_b{args.before:g}_a{args.after:g}"
    out_dir = resolve_path(args.out_root) / run_name
    db_path, ready, missing = prepare_sessions(args)

    all_rows, skipped_events, failed = [], 0, []
    for s, video_path in ready:
        try:
            rows, skipped = recut_session(s, video_path, out_dir, args.before, args.after, dry_run=args.dry_run)
        except Exception as e:
            failed.append({"sid": s["sid"], "video_id": s["video_id"], "reason": str(e)})
            continue
        all_rows.extend(rows)
        skipped_events += skipped
        print(f"{s['session_name']}: {len(rows)} clips" + (" (dry run)" if args.dry_run else ""))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db": repo_rel(db_path),
        "before_sec": args.before,
        "after_sec": args.after,
        "dry_run": bool(args.dry_run),
        "sessions_used": len(ready) - len(failed),
        "clips": len(all_rows),
        "clips_per_class": {c: sum(1 for r in all_rows if r["event_type"] == c) for c in EVENT_CLASSES},
        "events_skipped_other_type_or_empty": skipped_events,
        "sessions_skipped": missing + failed,
    }
    if not args.dry_run:
        write_rows_csv(all_rows, out_dir / "clips.csv", CLIP_COLUMNS)
        with open(out_dir / "recut_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "sessions_skipped"}, indent=2))
    for m in summary["sessions_skipped"]:
        print(f"SKIPPED {m['video_id']} ({m['sid'][:8]}): {m['reason']}")
    if not args.dry_run:
        print("Saved to:", repo_rel(out_dir))


if __name__ == "__main__":
    main()
