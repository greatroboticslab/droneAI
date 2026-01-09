# analysis/scripts/run_pipeline.py
import os
import csv
import time
import math
import argparse
from pathlib import Path

import cv2

from ultralytics import YOLO


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def read_manifest_csv(manifest_path: Path):
    """
    Expected columns (case-insensitive):
      - person
      - scenario   (Simulation / Real flight)
      - youtube_link
    """
    rows = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}

        required = ["person", "scenario", "youtube_link"]
        for r in required:
            if r not in cols:
                raise ValueError(f"manifest is missing required column: {r}")

        for row in reader:
            person = (row[cols["person"]] or "").strip()
            scenario = (row[cols["scenario"]] or "").strip()
            link = (row[cols["youtube_link"]] or "").strip()
            if person and scenario and link:
                rows.append({"person": person, "scenario": scenario, "youtube_link": link})
    return rows


def safe_filename(s: str):
    # Windows-safe-ish filename
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in s)
    return out.strip()[:120]


def download_with_ytdlp(url: str, out_dir: Path, label: str):
    """
    Downloads the best mp4 we can (fallbacks included). Returns file path or None.
    Requires: yt-dlp installed + ffmpeg accessible.
    """
    import yt_dlp

    ensure_dir(out_dir)
    outtmpl = str(out_dir / f"{safe_filename(label)}-%(id)s.%(ext)s")

    ydl_base = {
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
        # android client helps with SABR issues
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "retries": 5,
    }

    strategies = [
        "bv*+ba/bestvideo*+bestaudio",
        "best[ext=mp4]",
        "18",
        "best",
    ]

    for fmt in strategies:
        opts = dict(ydl_base)
        opts["format"] = fmt
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
                # Try to find the actual output
                # yt-dlp sometimes returns requested_downloads list
                if "requested_downloads" in info and info["requested_downloads"]:
                    for d in info["requested_downloads"]:
                        fp = d.get("filepath")
                        if fp and os.path.exists(fp):
                            return Path(fp)

                fp = info.get("_filename")
                if fp and os.path.exists(fp):
                    return Path(fp)

                # last-ditch guess by scanning out_dir for newest mp4
                mp4s = sorted(out_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
                if mp4s:
                    return mp4s[0]
        except Exception as e:
            print(f"[download] attempt fmt={fmt} failed: {e}")

    return None


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_sec = (total_frames / fps) if total_frames > 0 else 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return fps, int(total_frames), float(duration_sec), w, h


def predict_frame(model: YOLO, frame_bgr):
    """
    Returns:
      (label_str, score_float)
    Works for YOLO classification weights or detection weights.
    """
    results = model.predict(frame_bgr, verbose=False)

    if not results:
        return ("unknown", 0.0)

    r = results[0]

    # Classification case (common for best.pt in your folder)
    if hasattr(r, "probs") and r.probs is not None:
        top1 = int(r.probs.top1)
        conf = float(r.probs.top1conf)
        # model.names maps index->name
        name = model.names.get(top1, str(top1))
        return (str(name), conf)

    # Detection case fallback (count any detection as crash-ish)
    if hasattr(r, "boxes") and r.boxes is not None and len(r.boxes) > 0:
        # Pick the highest confidence detection
        confs = r.boxes.conf.cpu().numpy().tolist()
        best = max(confs) if confs else 0.0
        # If it detects any object we’ll call it "crash" (you can refine later)
        return ("crash", float(best))

    return ("unknown", 0.0)


def is_crash_label(label: str):
    """
    Heuristic mapping. Update once you know your model's real class names.
    """
    s = (label or "").strip().lower()
    # common possibilities
    crash_words = ["crash", "collision", "impact", "severe", "minor"]
    return any(w in s for w in crash_words)


def count_crash_events(per_frame, cooldown_sec=2.0, min_run_frames=2):
    """
    per_frame: list of dicts with keys {t, is_crash}
    We count an event when crash=True begins, and we enforce cooldown.
    """
    events = []
    last_event_t = -1e9
    run = 0
    in_crash = False

    for row in per_frame:
        t = row["t"]
        c = row["is_crash"]

        if c:
            run += 1
        else:
            run = 0

        if (not in_crash) and run >= min_run_frames:
            # start crash event
            if (t - last_event_t) >= cooldown_sec:
                events.append(t)
                last_event_t = t
            in_crash = True

        if in_crash and (not c):
            in_crash = False

    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to analysis/data/manifest.csv")
    parser.add_argument("--weights", required=True, help="Path to analysis/weights/best.pt")
    parser.add_argument("--out", default="analysis/output", help="Output folder")
    parser.add_argument("--downloads", default="analysis/output/downloads", help="Download cache folder")
    parser.add_argument("--sample_fps", type=float, default=2.0, help="Frame sampling rate (fps)")
    parser.add_argument("--conf", type=float, default=0.25, help="Min confidence threshold")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    weights_path = Path(args.weights)
    out_dir = Path(args.out)
    downloads_dir = Path(args.downloads)

    ensure_dir(out_dir)
    ensure_dir(downloads_dir)

    rows = read_manifest_csv(manifest_path)
    if not rows:
        raise SystemExit("No rows found in manifest.")

    model = YOLO(str(weights_path))

    per_frame_out = out_dir / "per_frame.csv"
    per_video_out = out_dir / "per_video.csv"

    per_frame_rows = []
    per_video_rows = []

    for i, item in enumerate(rows, start=1):
        person = item["person"]
        scenario = item["scenario"]
        url = item["youtube_link"]

        label = f"{person}-{scenario}"
        print(f"\n[{i}/{len(rows)}] Processing: {label}")

        video_path = download_with_ytdlp(url, downloads_dir, label=label)
        if not video_path or not video_path.exists():
            print("  ❌ Download failed.")
            per_video_rows.append({
                "person": person,
                "scenario": scenario,
                "youtube_link": url,
                "video_path": "",
                "duration_sec": 0.0,
                "crash_events": 0,
                "crashes_per_min": 0.0,
                "status": "download_failed",
            })
            continue

        info = get_video_info(video_path)
        if not info:
            print("  ❌ Could not open video.")
            per_video_rows.append({
                "person": person,
                "scenario": scenario,
                "youtube_link": url,
                "video_path": str(video_path),
                "duration_sec": 0.0,
                "crash_events": 0,
                "crashes_per_min": 0.0,
                "status": "open_failed",
            })
            continue

        fps, total_frames, duration_sec, w, h = info
        print(f"  ✅ Video: {video_path.name} | duration={duration_sec:.1f}s fps={fps:.2f}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("  ❌ OpenCV could not open video.")
            continue

        # sample every N frames to approximate sample_fps
        step = max(1, int(round(fps / max(args.sample_fps, 0.1))))
        frame_idx = 0

        local_frames = []
        while True:
            ok = cap.grab()
            if not ok:
                break

            if frame_idx % step == 0:
                ok2, frame = cap.retrieve()
                if not ok2 or frame is None:
                    frame_idx += 1
                    continue

                t = frame_idx / fps

                label_str, score = predict_frame(model, frame)
                crash = is_crash_label(label_str) and (score >= args.conf)

                row = {
                    "person": person,
                    "scenario": scenario,
                    "youtube_link": url,
                    "video_path": str(video_path),
                    "frame_idx": frame_idx,
                    "t_sec": round(t, 3),
                    "pred_label": label_str,
                    "pred_conf": round(score, 5),
                    "is_crash": int(crash),
                }
                per_frame_rows.append(row)
                local_frames.append({"t": t, "is_crash": crash})

            frame_idx += 1

        cap.release()

        # Count crash events from the per-frame crash booleans
        events = count_crash_events(local_frames, cooldown_sec=2.0, min_run_frames=2)
        crash_events = len(events)
        crashes_per_min = (crash_events / (duration_sec / 60.0)) if duration_sec > 0 else 0.0

        per_video_rows.append({
            "person": person,
            "scenario": scenario,
            "youtube_link": url,
            "video_path": str(video_path),
            "duration_sec": round(duration_sec, 3),
            "crash_events": crash_events,
            "crashes_per_min": round(crashes_per_min, 4),
            "status": "ok",
        })

        print(f"  📌 crash_events={crash_events} | crashes_per_min={crashes_per_min:.3f}")

    # Write outputs
    import pandas as pd
    pd.DataFrame(per_frame_rows).to_csv(per_frame_out, index=False)
    pd.DataFrame(per_video_rows).to_csv(per_video_out, index=False)

    print("\nDONE ✅")
    print("per-frame:", per_frame_out)
    print("per-video:", per_video_out)


if __name__ == "__main__":
    main()

