"""
Dataset import helpers: Excel rows and uploaded video files -> dataset_items.

Kept separate from app.py so it can be tested without starting Flask or MQTT.
"""
import hashlib
import os
import re
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm")

VIEW_TYPES = ["FPV", "Third-person", "Unknown"]
SOURCES = ["Real", "Simulation", "Unknown"]

_VIEW_ALIASES = {
    "fpv": "FPV",
    "first person": "FPV",
    "first-person": "FPV",
    "third person": "Third-person",
    "third-person": "Third-person",
    "thirdperson": "Third-person",
    "3rd person": "Third-person",
}
_SOURCE_ALIASES = {
    "real": "Real",
    "real flight": "Real",
    "sim": "Simulation",
    "simulation": "Simulation",
    "simulator": "Simulation",
}

_YOUTUBE_ID_RE = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})")


def normalize_view_type(value) -> str:
    text = str(value or "").strip()
    return _VIEW_ALIASES.get(text.lower(), "Unknown")


def normalize_source(value) -> str:
    text = str(value or "").strip()
    return _SOURCE_ALIASES.get(text.lower(), "Unknown")


def youtube_video_id(link: str) -> str:
    """The 11-char YouTube id, or '' if the link isn't a recognizable YouTube URL."""
    m = _YOUTUBE_ID_RE.search(link or "")
    return m.group(1) if m else ""


def _cell(row, col) -> str:
    if col is None:
        return ""
    value = row.get(col)
    if value is None or (isinstance(value, float) and value != value):  # NaN
        return ""
    return str(value).strip()


def _find_column(cols, candidates):
    for c in candidates:
        if c in cols:
            return cols[c]
    return None


def normalize_dataset_columns(df):
    """
    Returns (person_col, link_col, video_id_col, view_type_col, source_col).
    The last three are optional and may be None.
    """
    cols = {str(c).strip().lower(): c for c in df.columns}

    person_col = _find_column(cols, ["persona name", "person name", "person", "name"])
    link_col = _find_column(cols, ["youtube link", "youtube url", "youtube", "link", "url"])

    if not person_col or not link_col:
        raise ValueError("Excel must contain a person/name column and a YouTube link column.")

    video_id_col = _find_column(cols, ["video id", "video_id", "videoid"])
    view_type_col = _find_column(cols, ["view type", "view_type", "view"])
    source_col = _find_column(cols, ["source", "scenario"])

    return person_col, link_col, video_id_col, view_type_col, source_col


def build_dataset_items_from_excel(file_bytes: bytes, dataset_key: str):
    import pandas as pd

    df = pd.read_excel(BytesIO(file_bytes), sheet_name=0)
    person_col, link_col, video_id_col, view_type_col, source_col = normalize_dataset_columns(df)

    items = []
    row_num = 0
    for _, row in df.iterrows():
        person = _cell(row, person_col)
        link = _cell(row, link_col)
        if not link:
            continue
        row_num += 1

        # A stable id per source video, so train/test splits never put the
        # same video on both sides. YouTube id when there is one.
        video_id = _cell(row, video_id_col) or youtube_video_id(link)
        if not video_id:
            video_id = "url-" + hashlib.sha256(link.encode("utf-8")).hexdigest()[:10]

        items.append({
            "item_key": f"{dataset_key}:{row_num}",
            "row_index": row_num,
            "person_name": person,
            "youtube_link": link,
            "status": "not_labeled",
            "labeled_by": "",
            "locked_by": "",
            "scenario_type": "",
            "video_id": video_id,
            "view_type": normalize_view_type(_cell(row, view_type_col)),
            "source": normalize_source(_cell(row, source_col)),
            "local_path": "",
        })
    return items


def is_video_filename(filename: str) -> bool:
    return (filename or "").lower().endswith(VIDEO_EXTENSIONS)


def save_uploaded_video(file_storage, videos_dir: Path):
    """
    Save one uploaded video under videos_dir, named by a hash of its content.

    Returns (video_id, saved_path). The same file uploaded twice gets the same
    video_id and is stored once.
    """
    videos_dir = Path(videos_dir)
    videos_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file_storage.filename or "").suffix.lower() or ".mp4"
    sha = hashlib.sha256()

    fd, tmp_name = tempfile.mkstemp(dir=str(videos_dir), suffix=".part")
    try:
        with os.fdopen(fd, "wb") as out:
            while True:
                chunk = file_storage.stream.read(1024 * 1024)
                if not chunk:
                    break
                sha.update(chunk)
                out.write(chunk)

        video_id = "file-" + sha.hexdigest()[:12]
        final_path = videos_dir / f"{video_id}{ext}"
        if final_path.exists():
            os.remove(tmp_name)
        else:
            shutil.move(tmp_name, final_path)
        return video_id, final_path
    except Exception:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def build_video_file_item(dataset_key, row_index, video_id, saved_path, original_filename,
                          person_name, view_type, source, repo_dir: Path):
    person = (person_name or "").strip() or Path(original_filename or "").stem
    return {
        "item_key": f"{dataset_key}:{row_index}",
        "row_index": row_index,
        "person_name": person,
        # The queue shows this column; keep the original filename visible there.
        "youtube_link": original_filename or Path(saved_path).name,
        "status": "not_labeled",
        "labeled_by": "",
        "locked_by": "",
        "scenario_type": "",
        "video_id": video_id,
        "view_type": view_type if view_type in VIEW_TYPES else "Unknown",
        "source": source if source in SOURCES else "Unknown",
        "local_path": os.path.relpath(saved_path, start=str(repo_dir)),
    }
