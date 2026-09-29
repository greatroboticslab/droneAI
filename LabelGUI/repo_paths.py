"""
Repo-relative paths for the ML scripts.

Scripts store paths in their metadata and CSVs with repo_rel(), so a run made on
one machine (for example the Windows GPU PC) can be read on another. They read
paths back with resolve_path(), which also understands old absolute paths
written on another machine, such as C:\\Users\\name\\droneAI\\LabelGUI\\...
"""
import re
from pathlib import Path, PurePosixPath, PureWindowsPath

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELGUI_DIR = REPO_ROOT / "LabelGUI"

# Top-level folders of the repo. Used to find where the repo part of a foreign
# absolute path starts.
_REPO_TOP_DIRS = ("LabelGUI", "AI_Work", "analysis", "db", "docs")

_WINDOWS_ABS_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def _is_foreign_windows_abs(text: str) -> bool:
    return bool(_WINDOWS_ABS_RE.match(text)) and not Path(text).is_absolute()


def _parts(text: str):
    if _WINDOWS_ABS_RE.match(text):
        return PureWindowsPath(text).parts
    return PurePosixPath(text.replace("\\", "/")).parts


def _remap_into_repo(text: str):
    """C:\\Users\\x\\droneAI\\LabelGUI\\a\\b -> REPO_ROOT/LabelGUI/a/b, or None."""
    parts = _parts(text)
    for i, part in enumerate(parts):
        if part in _REPO_TOP_DIRS:
            return REPO_ROOT.joinpath(*parts[i:])
    return None


def resolve_path(value, *extra_bases, must_exist=False):
    """
    Turn a path from a CLI argument, CSV or JSON into a real Path on this machine.

    - Relative paths are tried against the current folder, the repo root,
      LabelGUI/, then extra_bases. The first one that exists wins.
    - Absolute paths that exist are used as they are.
    - Absolute paths from another machine (missing here) are remapped into this
      repo when they contain a repo folder name like LabelGUI.
    - When nothing exists, the repo-root guess is returned (or FileNotFoundError
      with must_exist=True), so error messages show a sensible path.

    Returns None for empty values (None, "", NaN).
    """
    if value is None:
        return None
    if isinstance(value, float) and value != value:  # NaN from pandas
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    if _is_foreign_windows_abs(text) or Path(text).is_absolute():
        own = Path(text)
        if own.is_absolute() and own.exists():
            return own.resolve()
        remapped = _remap_into_repo(text)
        if remapped is not None and (remapped.exists() or not must_exist):
            return remapped
        if must_exist:
            raise FileNotFoundError(text)
        return own

    rel = Path(text.replace("\\", "/"))
    candidates = [Path.cwd() / rel, REPO_ROOT / rel, LABELGUI_DIR / rel]
    candidates += [Path(b) / rel for b in extra_bases]
    for c in candidates:
        if c.exists():
            return c.resolve()
    if must_exist:
        raise FileNotFoundError(
            f"{text} (looked in: " + ", ".join(str(c.parent) for c in candidates) + ")"
        )
    return (REPO_ROOT / rel).resolve()


def repo_rel(path) -> str:
    """
    A path as a string to store in metadata: relative to the repo root with
    forward slashes when it is inside the repo, otherwise absolute (forward
    slashes). Empty string for None.
    """
    if path is None:
        return ""
    text = str(path)
    if not text:
        return ""
    if _is_foreign_windows_abs(text):
        remapped = _remap_into_repo(text)
        if remapped is None:
            return text.replace("\\", "/")
        p = remapped
    else:
        p = Path(text)
        if not p.is_absolute():
            p = Path.cwd() / p
    try:
        return p.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.resolve().as_posix()
