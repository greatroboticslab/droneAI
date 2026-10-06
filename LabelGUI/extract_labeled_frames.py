"""
Cut frames from the event clips of every labeled video, for the optical-flow step.

For each labeled video in the DB (newest finished session), reads the clips in
LabelGUI/ValidationResults/<session>/clips/ and writes
    LabelGUI/FrameDataset/<event>/<session>__<clip>__frame_000001.jpg
    LabelGUI/FrameDataset/frame_manifest.csv      (input of extract_dpflow_drone_features.py)

    python LabelGUI/extract_labeled_frames.py --snapshot snapshots/v3_2026-10-12   # for training runs
    python LabelGUI/extract_labeled_frames.py --fresh                              # live data
    python LabelGUI/extract_labeled_frames.py --dry-run

With --snapshot, the DB and clips come from that frozen snapshot (see
data_snapshot.py) and its id is recorded through optical flow and training, so
runs on different machines can be compared. Without it, the live DB is used.

Videos labeled with no events have no clips and are skipped. The DB is opened
read-only. --fresh deletes LabelGUI/FrameDataset first (including frames made from
the Frame Extraction page), so relabeled or removed events don't leave old frames.
"""
import argparse
import json
import shutil

from data_snapshot import read_snapshot, snapshot_ref
from repo_paths import repo_rel, resolve_path
from video_manifest import DEFAULT_DB, build_session_map, build_video_manifest


def select_sessions(manifest_rows, session_rows, view_type="all"):
    """Session folder names of the labeled videos, one per video (its labeled session)."""
    labeled_sids = {
        r["labeled_sid"] for r in manifest_rows
        if r.get("status") == "labeled" and r.get("labeled_sid")
        and (view_type == "all" or r.get("view_type") == view_type)
    }
    names = []
    for s in session_rows:
        if s["sid"] in labeled_sids and s["session_name"] not in names:
            names.append(s["session_name"])
    return names


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--snapshot", default="", help="Snapshot folder to read from (implies --fresh)")
    p.add_argument("--db", default=str(DEFAULT_DB), help="Labeling DB (default: db/droneai.sqlite)")
    p.add_argument("--view-type", choices=["all", "FPV", "Third-person"], default="all")
    p.add_argument("--sample-fps", type=float, default=5.0, help="Frames per second to keep (default 5)")
    p.add_argument("--fresh", action="store_true", help="Delete LabelGUI/FrameDataset first")
    p.add_argument("--dry-run", action="store_true", help="List the sessions, write nothing")
    args = p.parse_args()

    snapshot, results_dir = None, None
    if args.snapshot:
        snap_dir = resolve_path(args.snapshot, must_exist=True)
        snapshot = read_snapshot(snap_dir)
        if not snapshot:
            raise SystemExit(f"{args.snapshot} has no snapshot.json - is it an unzipped snapshot folder?")
        db_path = snap_dir / "droneai.sqlite"
        results_dir = snap_dir / "ValidationResults"
        args.fresh = True   # never mix frames from different data
        print(f"Using snapshot {snapshot['id']} ({snapshot['name']}, fingerprint {snapshot['fingerprint'][:12]})")
    else:
        db_path = resolve_path(args.db)
        print("Using the live DB (no snapshot): results can't be compared across machines.")
    manifest = build_video_manifest(db_path, active_only=True)
    sessions = build_session_map(db_path, manifest)
    names = select_sessions(manifest, sessions, args.view_type)
    labeled = sum(1 for r in manifest if r.get("status") == "labeled")
    print(f"Labeled videos in the active dataset: {labeled}; sessions to cut: {len(names)}")

    if args.dry_run:
        for n in names:
            print("  ", n)
        return

    # Imported here so --dry-run works without OpenCV.
    import frame_extraction_backend as fe

    if args.fresh and fe.FRAME_DATASET_DIR.exists():
        shutil.rmtree(fe.FRAME_DATASET_DIR)
        print("Removed", repo_rel(fe.FRAME_DATASET_DIR))

    # Which data these frames came from; optical flow and training pass it on.
    marker = fe.FRAME_DATASET_DIR / "snapshot.json"
    fe.FRAME_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    if snapshot:
        marker.write_text(json.dumps(snapshot_ref(snapshot), indent=2), encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    totals, skipped = {}, []
    for i, name in enumerate(names, 1):
        try:
            result = fe.extract_frames_from_session(name, sample_fps=args.sample_fps, results_dir=results_dir)
        except (FileNotFoundError, ValueError) as exc:
            skipped.append((name, str(exc)))
            continue
        for label, n in (result.get("label_counts") or {}).items():
            totals[label] = totals.get(label, 0) + n
        print(f"[{i}/{len(names)}] {name}")

    print("\nFrames per event:", totals)
    if skipped:
        print(f"Skipped {len(skipped)} sessions (usually labeled with no events):")
        for name, why in skipped:
            print(f"   {name}: {why}")
    print("Manifest:", repo_rel(fe.FRAME_DATASET_DIR / "frame_manifest.csv"))


if __name__ == "__main__":
    main()
