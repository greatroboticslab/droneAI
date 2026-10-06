"""
Frozen, numbered copies of the labeled data, so everyone trains on the same data.

A snapshot holds everything training needs and nothing that keeps changing:

    snapshots/v3_2026-10-12/
        snapshot.json              id, fingerprint, counts, who made it and when
        droneai.sqlite             copy of the labeling DB at that moment
        ValidationResults/<session>/clips/*.mp4   event clips of every labeled video
        video_manifest.csv         one row per video (from the snapshot's DB)
        session_videos.csv         session -> video_id, for grouped cross-validation
        test_videos.csv            the held-out test videos
    snapshots/v3_2026-10-12.zip    the same, in one file to download

The fingerprint is a hash of the labels (every event mark), the clips and the test
split. Two snapshots with the same fingerprint hold the same data, whatever their
name. Training runs record the snapshot id and fingerprint, and the dashboard
shows them.

    python LabelGUI/data_snapshot.py create          # on the labeling server
    python LabelGUI/data_snapshot.py list
    python LabelGUI/data_snapshot.py verify snapshots/v3_2026-10-12

Then, on any machine (after unzipping the snapshot into snapshots/):
    python LabelGUI/extract_labeled_frames.py --snapshot snapshots/v3_2026-10-12
"""
import argparse
import hashlib
import json
import platform
import shutil
import sqlite3
import zipfile
from datetime import datetime
from pathlib import Path

from repo_paths import LABELGUI_DIR, REPO_ROOT, repo_rel, resolve_path
from video_manifest import (
    COLUMNS,
    DEFAULT_DB,
    SESSION_COLUMNS,
    build_session_map,
    build_video_manifest,
    write_csv,
)

SNAPSHOTS_DIR = REPO_ROOT / "snapshots"
DEFAULT_TEST_SPLIT = REPO_ROOT / "analysis" / "data" / "test_videos.csv"
DEFAULT_RESULTS_DIR = LABELGUI_DIR / "ValidationResults"
EVENTS = ("takeoff", "land", "minor-crash", "severe-crash")


def labeled_sessions(manifest_rows, session_rows):
    """(session_name, sid) of the session that labeled each labeled video."""
    sids = {r["labeled_sid"] for r in manifest_rows if r.get("status") == "labeled" and r.get("labeled_sid")}
    out, seen = [], set()
    for s in session_rows:
        if s["sid"] in sids and s["session_name"] not in seen:
            seen.add(s["session_name"])
            out.append((s["session_name"], s["sid"]))
    return out


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _events_text(db_path, sids):
    """Every event mark of the given sessions, one sorted line each."""
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    try:
        rows = []
        for sid in sorted(sids):
            for idx, event_type, time_sec in conn.execute(
                    "SELECT idx, event_type, time_sec FROM validation_events WHERE sid=? ORDER BY idx", (sid,)):
                rows.append(f"{sid}\t{idx}\t{event_type}\t{float(time_sec):.3f}")
        return "\n".join(rows)
    finally:
        conn.close()


def fingerprint(snapshot_dir):
    """Hash of the labels, the clips and the test split (not of the DB file's bytes)."""
    snapshot_dir = Path(snapshot_dir)
    info = json.loads((snapshot_dir / "snapshot.json").read_text(encoding="utf-8"))
    h = hashlib.sha256()
    h.update(_events_text(snapshot_dir / "droneai.sqlite", info["session_sids"]).encode())
    files = sorted(snapshot_dir.glob("ValidationResults/*/clips/*.mp4")) + [snapshot_dir / "test_videos.csv"]
    for path in files:
        h.update(f"\n{path.relative_to(snapshot_dir).as_posix()}:{_sha256(path)}".encode())
    return h.hexdigest()


def _next_version(out_root):
    numbers = []
    for p in Path(out_root).glob("v*"):
        head = p.name.split("_")[0].split(".")[0][1:]
        if head.isdigit():
            numbers.append(int(head))
    return max(numbers, default=0) + 1


def create_snapshot(db_path=DEFAULT_DB, results_dir=DEFAULT_RESULTS_DIR, test_split=DEFAULT_TEST_SPLIT,
                    out_root=SNAPSHOTS_DIR, note="", make_zip=True):
    db_path, results_dir, test_split, out_root = map(Path, (db_path, results_dir, test_split, out_root))
    if not test_split.exists():
        raise FileNotFoundError(
            f"No test split at {repo_rel(test_split)}. Make it once with "
            "`python LabelGUI/make_test_split.py` before the first snapshot.")

    version = _next_version(out_root)
    snap_id = f"v{version}"
    snap_dir = out_root / f"{snap_id}_{datetime.now():%Y-%m-%d}"
    snap_dir.mkdir(parents=True)

    # A consistent copy even while the app is writing to the DB.
    src = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    dst = sqlite3.connect(snap_dir / "droneai.sqlite")
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()
    snap_db = snap_dir / "droneai.sqlite"

    manifest = build_video_manifest(snap_db, active_only=True)
    sessions = build_session_map(snap_db, manifest)
    write_csv(manifest, snap_dir / "video_manifest.csv", COLUMNS)
    write_csv(sessions, snap_dir / "session_videos.csv", SESSION_COLUMNS)
    shutil.copy2(test_split, snap_dir / "test_videos.csv")

    chosen = labeled_sessions(manifest, sessions)
    clips = 0
    for name, _ in chosen:
        src_clips = results_dir / name / "clips"
        if not src_clips.is_dir():
            continue
        for clip in sorted(src_clips.glob("*.mp4")):
            target = snap_dir / "ValidationResults" / name / "clips" / clip.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(clip, target)
            clips += 1

    labeled = [r for r in manifest if r.get("status") == "labeled"]
    info = {
        "id": snap_id,
        "name": snap_dir.name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "created_on": platform.node(),
        "note": note,
        "labeled_videos": len(labeled),
        "clips": clips,
        "events": {ev: sum(int(r.get("n_" + ev.replace("-", "_")) or 0) for r in labeled) for ev in EVENTS},
        "views": {v: sum(1 for r in labeled if r.get("view_type") == v)
                  for v in sorted({r.get("view_type") or "Unknown" for r in labeled})},
        "test_videos": max(0, len(test_split.read_text(encoding="utf-8").strip().splitlines()) - 1),
        "session_sids": [sid for _, sid in chosen],
    }
    (snap_dir / "snapshot.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    info["fingerprint"] = fingerprint(snap_dir)
    (snap_dir / "snapshot.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    if make_zip:
        zip_path = out_root / f"{snap_dir.name}.zip"
        # mp4 is already compressed; storing is as small and much faster.
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for path in sorted(snap_dir.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(out_root).as_posix())
    return snap_dir


def read_snapshot(snapshot_dir):
    path = Path(snapshot_dir) / "snapshot.json"
    try:
        info = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    info["dir"] = repo_rel(Path(snapshot_dir))
    zip_path = Path(snapshot_dir).with_suffix(".zip")
    info["zip"] = zip_path.name if zip_path.exists() else ""
    info["zip_mb"] = round(zip_path.stat().st_size / 1e6) if zip_path.exists() else None
    return info


def list_snapshots(out_root=SNAPSHOTS_DIR):
    """All snapshots, newest version first."""
    found = [read_snapshot(p) for p in Path(out_root).glob("v*") if p.is_dir()]
    found = [s for s in found if s]
    return sorted(found, key=lambda s: int(s["id"][1:]) if s["id"][1:].isdigit() else 0, reverse=True)


def snapshot_ref(info):
    """What a training run records about the data it used."""
    return {"id": info["id"], "name": info["name"], "fingerprint": info["fingerprint"],
            "dir": info.get("dir") or ""}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="Freeze the current labeled data")
    c.add_argument("--db", default=str(DEFAULT_DB))
    c.add_argument("--test-split", default=str(DEFAULT_TEST_SPLIT))
    c.add_argument("--note", default="", help="One line about this snapshot, e.g. 'after Holly's batch'")
    c.add_argument("--no-zip", action="store_true")
    sub.add_parser("list", help="Show the snapshots on this machine")
    v = sub.add_parser("verify", help="Check a snapshot still matches its fingerprint")
    v.add_argument("snapshot_dir")
    args = p.parse_args()

    if args.cmd == "create":
        snap = create_snapshot(resolve_path(args.db, must_exist=True), test_split=resolve_path(args.test_split),
                               note=args.note, make_zip=not args.no_zip)
        info = read_snapshot(snap)
        print(f"Snapshot {info['id']} -> {info['dir']}")
        print(f"  {info['labeled_videos']} labeled videos, {info['clips']} clips, events {info['events']}")
        print(f"  fingerprint {info['fingerprint'][:12]}")
        if info["zip"]:
            print(f"  zip: snapshots/{info['zip']} ({info['zip_mb']} MB)")
    elif args.cmd == "list":
        snaps = list_snapshots()
        if not snaps:
            print("No snapshots yet. Make one with: python LabelGUI/data_snapshot.py create")
        for s in snaps:
            print(f"{s['id']:>4}  {s['name']:<18} {s['created_at']}  {s['labeled_videos']:>4} videos "
                  f"{s['clips']:>5} clips  {s['fingerprint'][:12]}  {s.get('note', '')}")
    else:
        snap = resolve_path(args.snapshot_dir, must_exist=True)
        info = read_snapshot(snap)
        actual = fingerprint(snap)
        if actual == info["fingerprint"]:
            print(f"OK: {info['id']} matches its fingerprint {actual[:12]}")
        else:
            raise SystemExit(f"CHANGED: {info['id']} was {info['fingerprint'][:12]}, files now give {actual[:12]}")


if __name__ == "__main__":
    main()
