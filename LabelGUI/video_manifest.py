"""
Export a video manifest from the labeling DB, one row per video, for the ML scripts.

    python LabelGUI/video_manifest.py                      # all datasets
    python LabelGUI/video_manifest.py --active-only        # only the active dataset
    python LabelGUI/video_manifest.py --out some/file.csv

Default output: analysis/data/video_manifest.csv. The DB is opened read-only.

Columns
    video_id         stable id per source video (YouTube id, file-<hash>, or url-<hash>)
    view_type        FPV / Third-person / Unknown
    source           Real / Simulation / Unknown
    link             YouTube (or other) link; empty for uploaded files
    local_path       repo-relative path of an uploaded video file; empty for links
    file_name        original file name of an uploaded video; empty for links
    status           labeled / in_progress / not_labeled
    person_name      the pilot / persona from the dataset
    labeled_by       who finished labeling it
    dataset_name     dataset the row comes from
    item_key         the dataset row (dataset_key:row) that was kept
    item_count       how many dataset rows point at this video (duplicates are merged)
    final_sessions   finished labeling sessions for this video
    labeled_sid      the newest finished session, whose events are counted below
    events_total     events in that session
    n_takeoff, n_land, n_minor_crash, n_severe_crash, n_other
                     events per class in that session
    updated_at       last change of the kept dataset row
"""
import argparse
import csv
import hashlib
import sqlite3
from pathlib import Path

from dataset_import import normalize_source, normalize_view_type, youtube_video_id
from repo_paths import REPO_ROOT, repo_rel, resolve_path

DEFAULT_DB = REPO_ROOT / "db" / "droneai.sqlite"
DEFAULT_OUT = REPO_ROOT / "analysis" / "data" / "video_manifest.csv"

EVENT_CLASSES = ("takeoff", "land", "minor-crash", "severe-crash")

COLUMNS = [
    "video_id", "view_type", "source", "link", "local_path", "file_name", "status",
    "person_name", "labeled_by", "dataset_name", "item_key", "item_count",
    "final_sessions", "labeled_sid", "events_total",
    "n_takeoff", "n_land", "n_minor_crash", "n_severe_crash", "n_other", "updated_at",
]

# When one video is in several dataset rows, keep the most advanced one.
_STATUS_RANK = {"labeled": 2, "in_progress": 1, "not_labeled": 0}


def _connect_read_only(db_path: Path):
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def link_key(link: str) -> str:
    """Key that matches a dataset row to its sessions: YouTube id, else the link text."""
    link = (link or "").strip()
    return youtube_video_id(link) or link


def item_video_id(item) -> str:
    """Same rule as the Excel importer, for rows imported before video_id existed."""
    vid = (item.get("video_id") or "").strip()
    if vid:
        return vid
    link = (item.get("youtube_link") or "").strip()
    return youtube_video_id(link) or ("url-" + hashlib.sha256(link.encode("utf-8")).hexdigest()[:10])


_ITEMS_SQL = """
    SELECT i.*, d.dataset_name AS dataset_name
    FROM dataset_items i
    LEFT JOIN datasets d ON d.dataset_key = i.dataset_key
    WHERE (? = 0 OR d.is_active = 1)
    ORDER BY i.dataset_key, i.row_index
"""


def _load_items(conn, active_only):
    items = []
    for r in conn.execute(_ITEMS_SQL, (1 if active_only else 0,)):
        item = dict(r)
        # DBs made before these columns existed simply don't have them.
        for c in ("video_id", "view_type", "source", "local_path", "labeled_by", "updated_at"):
            item[c] = item.get(c) or ""
        items.append(item)
    return items


def _load_final_sessions(conn):
    """{link_key: [sessions newest first]} for sessions that finished normally."""
    sessions = {}
    rows = conn.execute(
        "SELECT sid, youtube_link, created_at, updated_at FROM validation_sessions "
        "WHERE status = 'final' ORDER BY COALESCE(updated_at, created_at, '') DESC"
    )
    for r in rows:
        sessions.setdefault(link_key(r["youtube_link"]), []).append(dict(r))
    return sessions


def _event_counts(conn, sid):
    counts = {c: 0 for c in EVENT_CLASSES}
    other = 0
    for r in conn.execute(
        "SELECT event_type, COUNT(*) AS n FROM validation_events WHERE sid = ? GROUP BY event_type", (sid,)
    ):
        if r["event_type"] in counts:
            counts[r["event_type"]] = r["n"]
        else:
            other += r["n"]
    return counts, other


def _pick(items):
    """The dataset row to keep for one video, plus metadata filled from the others."""
    best = max(items, key=lambda it: (_STATUS_RANK.get(it["status"], -1), it["updated_at"]))
    view = normalize_view_type(best["view_type"])
    source = normalize_source(best["source"])
    # Fill Unknown from a duplicate row that has the value.
    for it in items:
        if view == "Unknown":
            view = normalize_view_type(it["view_type"])
        if source == "Unknown":
            source = normalize_source(it["source"])
    return best, view, source


def build_video_manifest(db_path, active_only=False):
    """Rows (dicts with COLUMNS) for every video in the DB, one per video_id."""
    conn = _connect_read_only(db_path)
    try:
        items = _load_items(conn, active_only)
        sessions = _load_final_sessions(conn)

        by_video = {}
        for it in items:
            by_video.setdefault(item_video_id(it), []).append(it)

        rows = []
        for vid, group in by_video.items():
            best, view, source = _pick(group)
            is_file = bool(best["local_path"])
            link = best["youtube_link"] or ""

            finals = []
            for it in group:
                for s in sessions.get(link_key(it["youtube_link"]), []):
                    if s not in finals:
                        finals.append(s)
            finals.sort(key=lambda s: s.get("updated_at") or s.get("created_at") or "", reverse=True)

            counts, other = ({c: 0 for c in EVENT_CLASSES}, 0)
            labeled_sid = ""
            if finals:
                labeled_sid = finals[0]["sid"]
                counts, other = _event_counts(conn, labeled_sid)

            rows.append({
                "video_id": vid,
                "view_type": view,
                "source": source,
                "link": "" if is_file else link,
                "local_path": repo_rel(REPO_ROOT / best["local_path"]) if is_file else "",
                "file_name": link if is_file else "",
                "status": best["status"] or "not_labeled",
                "person_name": best["person_name"] or "",
                "labeled_by": best["labeled_by"] or "",
                "dataset_name": best["dataset_name"] or "",
                "item_key": best["item_key"],
                "item_count": len(group),
                "final_sessions": len(finals),
                "labeled_sid": labeled_sid,
                "events_total": sum(counts.values()) + other,
                "n_takeoff": counts["takeoff"],
                "n_land": counts["land"],
                "n_minor_crash": counts["minor-crash"],
                "n_severe_crash": counts["severe-crash"],
                "n_other": other,
                "updated_at": best["updated_at"],
            })
        return rows
    finally:
        conn.close()


def write_video_manifest(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Export one row per video from the labeling DB.")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite DB (default: db/droneai.sqlite)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="CSV to write (default: analysis/data/video_manifest.csv)")
    parser.add_argument("--active-only", action="store_true", help="Only the active dataset")
    args = parser.parse_args()

    db_path = resolve_path(args.db, must_exist=True)
    rows = build_video_manifest(db_path, active_only=args.active_only)
    out = write_video_manifest(rows, resolve_path(args.out))

    labeled = sum(1 for r in rows if r["status"] == "labeled")
    print(f"Wrote {len(rows)} videos ({labeled} labeled) to {repo_rel(out)}")
    totals = {c: sum(r[c] for r in rows) for c in ("n_takeoff", "n_land", "n_minor_crash", "n_severe_crash")}
    print("Events in finished sessions:", ", ".join(f"{k[2:]}={v}" for k, v in totals.items()))


if __name__ == "__main__":
    main()
