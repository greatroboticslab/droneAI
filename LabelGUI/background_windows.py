"""
Cut "nothing happening" (background) windows from fully labeled videos.

A finished labeling session has every take-off, landing and crash marked up to
where playback stopped (watched_until_sec: the end, or earlier when the labeler
used "finish early"), so the rest of that watched part is background. Sessions
from before watched_until_sec was recorded are treated as watched to the end.
Windows are kept at least --gap seconds away from every event mark, and are the
same length as the event clips by default (-2 s / +10 s = 12 s), so a 5-class
model (4 events + background) sees inputs of the same length.

    python LabelGUI/background_windows.py --dry-run
    python LabelGUI/background_windows.py                         # 12 s windows, 3 s gap, 5 per video
    python LabelGUI/background_windows.py --length 3 --per-video 20 --run-name bg_3s

Output (default LabelGUI/RecutClips/<run-name>/), same layout as the event clips:
    <session_name>/clips/bg_<nn>.mp4
    windows.csv                 one row per window (event_type = background)
    background_summary.json

Only the newest finished session per video is used, unless --all-sessions.
If a labeler stopped early, that video is not fully labeled: leave it out with
--exclude-videos (CSV with a video_id column).
"""
import argparse
import json
from datetime import datetime

import pandas as pd

from clip_dataset import (
    add_common_args,
    cut_segments,
    open_video,
    prepare_sessions,
    write_rows_csv,
)
from repo_paths import repo_rel, resolve_path

WINDOW_COLUMNS = [
    "clip_path", "video_id", "view_type", "source", "session_name", "sid", "window_idx",
    "event_type", "clip_start_sec", "clip_end_sec", "length_sec", "gap_sec",
    "nearest_event_sec", "watched_until_sec", "fps", "frames", "video_path",
]


def free_intervals(duration, event_times, gap):
    """Parts of [0, duration] that are at least `gap` seconds from every event time."""
    blocked = sorted((max(0.0, t - gap), min(duration, t + gap)) for t in event_times)
    free, cursor = [], 0.0
    for start, end in blocked:
        if start > cursor:
            free.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        free.append((cursor, duration))
    return free


def background_windows(duration, event_times, length, gap, max_windows):
    """
    Up to max_windows non-overlapping windows of `length` seconds, each fully
    inside a free interval, spread evenly over the video. Returns [(start, end)].
    """
    if duration <= 0 or length <= 0 or max_windows <= 0:
        return []
    candidates = []
    for a, b in free_intervals(duration, event_times, gap):
        n = int((b - a) // length)
        if n <= 0:
            continue
        # Center the tiles inside the free interval.
        pad = ((b - a) - n * length) / 2.0
        candidates += [(a + pad + i * length, a + pad + (i + 1) * length) for i in range(n)]
    if len(candidates) <= max_windows:
        return candidates
    # Evenly spaced pick, always including the first and the last candidate.
    if max_windows == 1:
        return [candidates[len(candidates) // 2]]
    step = (len(candidates) - 1) / (max_windows - 1)
    return [candidates[round(i * step)] for i in range(max_windows)]


def usable_duration(video_duration, session):
    """
    Seconds of the video that were labeled: up to watched_until_sec when it was
    recorded, else the whole video. Returns (seconds, watch_time_was_recorded).
    """
    duration = video_duration or session.get("duration_sec") or 0.0
    watched = session.get("watched_until_sec") or 0.0
    if watched > 0:
        return (min(duration, watched) if duration > 0 else watched), True
    return duration, False


def nearest_event_distance(start, end, event_times):
    if not event_times:
        return ""
    d = min(max(start - t, t - end, 0.0) for t in event_times)
    return round(d, 3)


def main():
    parser = argparse.ArgumentParser(description="Cut background windows from fully labeled videos.")
    parser.add_argument("--length", type=float, default=12.0,
                        help="Window length in seconds (default 12 = the -2 s / +10 s event clip)")
    parser.add_argument("--gap", type=float, default=3.0, help="Minimum seconds from any event (default 3)")
    parser.add_argument("--per-video", type=int, default=5, help="Maximum windows per video (default 5)")
    parser.add_argument("--exclude-videos", default="",
                        help="CSV with a video_id column: videos that were not fully labeled")
    add_common_args(parser, default_run="")
    args = parser.parse_args()
    if args.length <= 0 or args.gap < 0 or args.per_video <= 0:
        parser.error("--length and --per-video must be > 0, --gap >= 0")

    run_name = args.run_name or f"background_l{args.length:g}_g{args.gap:g}"
    out_dir = resolve_path(args.out_root) / run_name

    excluded = set()
    if args.exclude_videos:
        excluded = set(pd.read_csv(resolve_path(args.exclude_videos, must_exist=True), dtype=str)["video_id"].dropna())

    db_path, ready, missing = prepare_sessions(args)
    rows, failed = [], []
    no_watch_time = 0
    for s, video_path in ready:
        if s["video_id"] in excluded:
            continue
        times = [t for _, _, t in s["events"]]  # every mark counts, whatever its type
        cap, fps, duration = (None, 0.0, 0.0)
        if not args.dry_run:
            cap, fps, duration = open_video(video_path)
            if cap is None:
                failed.append({"sid": s["sid"], "video_id": s["video_id"], "reason": f"cannot open {video_path}"})
                continue
        duration, recorded = usable_duration(duration, s)
        watched = s.get("watched_until_sec") or 0.0
        if not recorded:
            no_watch_time += 1
        if duration <= 0:
            failed.append({"sid": s["sid"], "video_id": s["video_id"], "reason": "unknown video duration"})
            if cap is not None:
                cap.release()
            continue

        windows = background_windows(duration, times, args.length, args.gap, args.per_video)
        paths = [out_dir / s["session_name"] / "clips" / f"bg_{k:02d}.mp4" for k in range(len(windows))]
        frames = [0] * len(windows)
        if cap is not None:
            try:
                frames = cut_segments(cap, fps, [(a, b, p) for (a, b), p in zip(windows, paths)])
            finally:
                cap.release()

        for k, ((a, b), p, n) in enumerate(zip(windows, paths, frames)):
            if not args.dry_run and n == 0:
                continue
            rows.append({
                "clip_path": repo_rel(p),
                "video_id": s["video_id"],
                "view_type": s["view_type"],
                "source": s["source"],
                "session_name": s["session_name"],
                "sid": s["sid"],
                "window_idx": k,
                "event_type": "background",
                "clip_start_sec": round(a, 3),
                "clip_end_sec": round(b, 3),
                "length_sec": args.length,
                "gap_sec": args.gap,
                "nearest_event_sec": nearest_event_distance(a, b, times),
                "watched_until_sec": round(watched, 3) if watched > 0 else "",
                "fps": round(fps, 3),
                "frames": n,
                "video_path": repo_rel(video_path) if video_path else "",
            })
        print(f"{s['session_name']}: {len(windows)} background windows" + (" (dry run)" if args.dry_run else ""))

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "db": repo_rel(db_path),
        "length_sec": args.length,
        "gap_sec": args.gap,
        "per_video": args.per_video,
        "dry_run": bool(args.dry_run),
        "videos_excluded": len(excluded),
        "sessions_without_watched_time": no_watch_time,
        "windows": len(rows),
        "videos_with_windows": len({r["video_id"] for r in rows}),
        "sessions_skipped": missing + failed,
    }
    if not args.dry_run:
        write_rows_csv(rows, out_dir / "windows.csv", WINDOW_COLUMNS)
        with open(out_dir / "background_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "sessions_skipped"}, indent=2))
    for m in summary["sessions_skipped"]:
        print(f"SKIPPED {m['video_id']} ({m['sid'][:8]}): {m['reason']}")
    if not args.dry_run:
        print("Saved to:", repo_rel(out_dir))


if __name__ == "__main__":
    main()
