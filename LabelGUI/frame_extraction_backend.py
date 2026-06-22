import os
import re
import cv2
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
VALIDATION_RESULTS_DIR = BASE_DIR / "ValidationResults"
FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"


def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r'[<>:"/\\|?*]+', "_", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name)
    name = name.strip("._ ")
    return name or "unnamed"


def _is_safe_child(base: Path, target: Path) -> bool:
    try:
        base = base.resolve()
        target = target.resolve()
        return str(target).startswith(str(base))
    except Exception:
        return False


def parse_label_from_clip_name(filename: str) -> str:
    """
    Supports names like:
      001_takeoff.mp4
      002_minor-crash.mp4
      takeoff_01.mp4
      minor-crash_02.mp4
    """
    stem = Path(filename).stem

    # New format: 001_takeoff
    m = re.match(r"^\d+_(.+)$", stem)
    if m:
        return _safe_name(m.group(1).lower())

    # Old format: takeoff_01
    m = re.match(r"^(.+)_\d+$", stem)
    if m:
        return _safe_name(m.group(1).lower())

    return _safe_name(stem.lower())


def list_validation_sessions():
    """
    Returns session folders inside ValidationResults that contain clips/*.mp4.
    """
    VALIDATION_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    sessions = []

    for folder in sorted(VALIDATION_RESULTS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not folder.is_dir():
            continue

        clips_dir = folder / "clips"
        if not clips_dir.exists() or not clips_dir.is_dir():
            continue

        clips = sorted([p for p in clips_dir.iterdir() if p.suffix.lower() == ".mp4"])

        if not clips:
            continue

        metadata = {}
        metadata_path = folder / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    metadata = json.load(f)
            except Exception:
                metadata = {}

        labels = sorted(list({parse_label_from_clip_name(c.name) for c in clips}))

        sessions.append({
            "name": folder.name,
            "path": str(folder),
            "clip_count": len(clips),
            "labels": labels,
            "person_name": metadata.get("person_name", ""),
            "scenario_base": metadata.get("scenario_base", ""),
            "youtube_link": metadata.get("youtube_link", ""),
            "created_at": metadata.get("created_at", ""),
        })

    return sessions

def _make_unique_source_name(base_name: str) -> str:
    """
    Makes sure each uploaded clip folder gets a unique source name.
    This prevents one upload from deleting or overwriting another upload.
    """
    base_name = _safe_name(base_name or "uploaded_clip_folder")

    existing = set()

    global_manifest_path = FRAME_DATASET_DIR / "frame_manifest.csv"

    if global_manifest_path.exists():
        try:
            old_df = pd.read_csv(global_manifest_path)
            if "source_name" in old_df.columns:
                existing.update(old_df["source_name"].dropna().astype(str).tolist())
            elif "session_name" in old_df.columns:
                existing.update(old_df["session_name"].dropna().astype(str).tolist())
        except Exception:
            pass

    if FRAME_DATASET_DIR.exists():
        for label_dir in FRAME_DATASET_DIR.iterdir():
            if not label_dir.is_dir():
                continue

            for img in label_dir.glob("*.jpg"):
                stem = img.stem
                if "__" in stem:
                    existing.add(stem.split("__")[0])

    if base_name not in existing:
        return base_name

    counter = 2

    while True:
        candidate = f"{base_name}_{counter}"

        if candidate not in existing:
            return candidate

        counter += 1


def get_session_clips(session_name: str):
    """
    Returns clips for one validation session.
    """
    session_name = _safe_name(session_name)
    session_dir = VALIDATION_RESULTS_DIR / session_name
    clips_dir = session_dir / "clips"

    if not _is_safe_child(VALIDATION_RESULTS_DIR, session_dir):
        return []

    if not clips_dir.exists():
        return []

    clips = []
    for clip in sorted(clips_dir.iterdir()):
        if clip.suffix.lower() != ".mp4":
            continue

        label = parse_label_from_clip_name(clip.name)

        duration_sec = 0.0
        frame_count = 0
        fps = 0.0

        cap = cv2.VideoCapture(str(clip))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if fps:
                duration_sec = frame_count / fps
        cap.release()

        clips.append({
            "filename": clip.name,
            "label": label,
            "duration_sec": round(duration_sec, 3),
            "frame_count": frame_count,
            "fps": round(fps, 3),
        })

    return clips


def extract_frames_from_session(session_name: str, sample_fps: float = 5.0, overwrite: bool = True):
    """
    Extracts frames from all clips in one validation session.

    Output:
      LabelGUI/FrameDataset/<label>/<session>__<clip>__frame_000001.jpg
    """
    session_name = _safe_name(session_name)
    sample_fps = float(sample_fps or 5.0)

    if sample_fps <= 0:
        sample_fps = 5.0

    session_dir = VALIDATION_RESULTS_DIR / session_name
    clips_dir = session_dir / "clips"

    if not _is_safe_child(VALIDATION_RESULTS_DIR, session_dir):
        raise ValueError("Invalid session folder.")

    if not clips_dir.exists():
        raise FileNotFoundError(f"No clips folder found for session: {session_name}")

    FRAME_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    clips = sorted([p for p in clips_dir.iterdir() if p.suffix.lower() == ".mp4"])

    if not clips:
        raise FileNotFoundError("No .mp4 clips found in this session.")

    manifest_rows = []
    label_counts = {}
    preview_images = []

    extracted_at = datetime.utcnow().isoformat()

    for clip_path in clips:
        label = parse_label_from_clip_name(clip_path.name)
        label_dir = FRAME_DATASET_DIR / label
        label_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(clip_path))
        if not cap.isOpened():
            continue

        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Extract one frame every N frames.
        step = max(1, int(round(source_fps / sample_fps)))

        frame_index = 0
        saved_index = 0

        clip_stem = _safe_name(clip_path.stem)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_index % step == 0:
                saved_index += 1

                out_name = f"{session_name}__{clip_stem}__frame_{saved_index:06d}.jpg"
                out_path = label_dir / out_name

                if overwrite or not out_path.exists():
                    cv2.imwrite(str(out_path), frame)

                rel_path = out_path.relative_to(FRAME_DATASET_DIR)

                frame_time_sec = frame_index / source_fps if source_fps else 0.0

                manifest_rows.append({
                    "session_name": session_name,
                    "clip_filename": clip_path.name,
                    "label": label,
                    "source_frame_index": frame_index,
                    "frame_time_sec": round(frame_time_sec, 3),
                    "image_path": str(rel_path).replace("\\", "/"),
                    "extracted_at": extracted_at,
                })

                label_counts[label] = label_counts.get(label, 0) + 1

                if len(preview_images) < 12:
                    preview_images.append({
                        "label": label,
                        "path": str(rel_path).replace("\\", "/"),
                    })

            frame_index += 1

        cap.release()

    if manifest_rows:
        manifest_df = pd.DataFrame(manifest_rows)

        # Per-session manifest
        session_manifest_path = FRAME_DATASET_DIR / f"{session_name}_frame_manifest.csv"
        manifest_df.to_csv(session_manifest_path, index=False)

        # Global manifest append/update style
        global_manifest_path = FRAME_DATASET_DIR / "frame_manifest.csv"
        if global_manifest_path.exists():
            try:
                old_df = pd.read_csv(global_manifest_path)
                combined = pd.concat([old_df, manifest_df], ignore_index=True)
                combined = combined.drop_duplicates(subset=["image_path"], keep="last")
                combined.to_csv(global_manifest_path, index=False)
            except Exception:
                manifest_df.to_csv(global_manifest_path, index=False)
        else:
            manifest_df.to_csv(global_manifest_path, index=False)

    summary = {
        "session_name": session_name,
        "sample_fps": sample_fps,
        "clips_processed": len(clips),
        "total_frames_saved": len(manifest_rows),
        "label_counts": label_counts,
        "output_dir": str(FRAME_DATASET_DIR),
        "preview_images": preview_images,
    }

    summary_path = FRAME_DATASET_DIR / f"{session_name}_extraction_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary

def extract_frames_from_uploaded_clip_folder(uploaded_files, source_name: str = "", sample_fps: float = 5.0, overwrite: bool = True):
    """
    Extract frames directly from an uploaded folder of clips.

    Every uploaded folder is treated as a new source.
    It does NOT delete frames from previous uploaded folders.

    Expected clip names:
        001_takeoff.mp4
        002_land.mp4
        003_minor-crash.mp4
        004_severe-crash.mp4
    """
    sample_fps = float(sample_fps or 5.0)

    if sample_fps <= 0:
        sample_fps = 5.0

    FRAME_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Keep only mp4 files
    mp4_files = []

    for f in uploaded_files:
        filename = (f.filename or "").replace("\\", "/")

        if filename.lower().endswith(".mp4"):
            mp4_files.append(f)

    if not mp4_files:
        raise FileNotFoundError("No .mp4 clips were found in the uploaded folder.")

    # If user did not type a source name, infer it from the uploaded folder name.
    if not source_name:
        first_name = (mp4_files[0].filename or "").replace("\\", "/")
        parts = [p for p in first_name.split("/") if p]

        if len(parts) > 1:
            source_name = parts[0]
        else:
            source_name = "uploaded_clip_folder"

    # Make every upload unique so uploads do not replace each other.
    source_name = _make_unique_source_name(source_name)

    extracted_at = datetime.utcnow().isoformat()

    manifest_rows = []
    label_counts = {}
    preview_images = []

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        for uploaded_file in mp4_files:
            original_rel_path = (uploaded_file.filename or "").replace("\\", "/")
            original_name = Path(original_rel_path).name

            safe_clip_name = _safe_name(original_name)

            temp_clip_path = temp_dir / safe_clip_name
            uploaded_file.save(str(temp_clip_path))

            label = parse_label_from_clip_name(original_name)
            label = _safe_name(label.lower()).replace("_", "-")

            label_dir = FRAME_DATASET_DIR / label
            label_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(temp_clip_path))

            if not cap.isOpened():
                continue

            source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            step = max(1, int(round(source_fps / sample_fps)))

            frame_index = 0
            saved_index = 0

            clip_stem = _safe_name(
                Path(original_rel_path).with_suffix("").as_posix().replace("/", "__")
            )

            while True:
                ret, frame = cap.read()

                if not ret:
                    break

                if frame_index % step == 0:
                    saved_index += 1

                    out_name = f"{source_name}__{clip_stem}__frame_{saved_index:06d}.jpg"
                    out_path = label_dir / out_name

                    if overwrite or not out_path.exists():
                        cv2.imwrite(str(out_path), frame)

                    rel_path = out_path.relative_to(FRAME_DATASET_DIR)
                    rel_path_str = str(rel_path).replace("\\", "/")

                    frame_time_sec = frame_index / source_fps if source_fps else 0.0

                    manifest_rows.append({
                        "source_type": "uploaded_clip_folder",
                        "source_name": source_name,
                        "session_name": source_name,
                        "clip_filename": original_rel_path,
                        "label": label,
                        "source_frame_index": frame_index,
                        "frame_time_sec": round(frame_time_sec, 3),
                        "image_path": rel_path_str,
                        "extracted_at": extracted_at,
                    })

                    label_counts[label] = label_counts.get(label, 0) + 1

                    if len(preview_images) < 12:
                        preview_images.append({
                            "label": label,
                            "path": rel_path_str,
                        })

                frame_index += 1

            cap.release()

    if not manifest_rows:
        raise RuntimeError("The clips were uploaded, but no frames were extracted.")

    new_df = pd.DataFrame(manifest_rows)

    safe_source = _safe_name(source_name)

    # Save per-upload manifest
    source_manifest_path = FRAME_DATASET_DIR / f"{safe_source}_frame_manifest.csv"
    new_df.to_csv(source_manifest_path, index=False)

    # Append to global manifest used by ViT training.
    # Do not delete previous uploaded folders.
    global_manifest_path = FRAME_DATASET_DIR / "frame_manifest.csv"

    if global_manifest_path.exists():
        try:
            old_df = pd.read_csv(global_manifest_path)
            combined = pd.concat([old_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["image_path"], keep="last")
            combined.to_csv(global_manifest_path, index=False)
        except Exception:
            new_df.to_csv(global_manifest_path, index=False)
    else:
        new_df.to_csv(global_manifest_path, index=False)

    summary = {
        "source_type": "uploaded_clip_folder",
        "source_name": source_name,
        "session_name": source_name,
        "sample_fps": sample_fps,
        "clips_processed": len(mp4_files),
        "total_frames_saved": len(manifest_rows),
        "label_counts": label_counts,
        "output_dir": str(FRAME_DATASET_DIR),
        "preview_images": preview_images,
    }

    summary_path = FRAME_DATASET_DIR / f"{safe_source}_extraction_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary
