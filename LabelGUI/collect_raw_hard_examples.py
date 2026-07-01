import argparse
import csv
import shutil
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_RAW_ROOTS = [
    BASE_DIR / "FrameDataset",
    BASE_DIR / "RoboflowDetectionUpload",
]

DEFAULT_OUTPUT_BASE = BASE_DIR / "RoboflowDetectionUpload"


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_stem(stem: str) -> str:
    """
    Handles possible overlay naming variants.

    Example:
      image_name_with_box.jpg -> image_name
      image_name_overlay.jpg  -> image_name

    Our current overlay script keeps the exact same name, but this makes it safer.
    """
    suffixes = [
        "_with_box",
        "_overlay",
        "_prediction",
        "_pred",
        "_detected",
    ]

    changed = True

    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True

    return stem


def collect_images(folder: Path):
    folder = Path(folder)

    images = []

    for path in folder.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            images.append(path)

    return sorted(images)


def build_raw_index(raw_roots):
    """
    Builds two indexes:
      1. exact filename match
      2. stem match without extension

    This makes it easier to find raw images even if file extensions differ.
    """
    by_name = {}
    by_stem = {}

    total = 0

    for root in raw_roots:
        root = Path(root)

        if not root.exists():
            continue

        for path in root.rglob("*"):
            if not path.is_file():
                continue

            if path.suffix.lower() not in IMAGE_EXTS:
                continue

            total += 1

            by_name.setdefault(path.name, []).append(path)
            by_stem.setdefault(path.stem, []).append(path)

    return by_name, by_stem, total


def choose_best_match(matches, overlay_name):
    """
    If multiple raw images have the same filename, choose one.
    Usually there should only be one.
    """
    if not matches:
        return None

    # Prefer FrameDataset over other folders if available.
    frame_matches = [m for m in matches if "FrameDataset" in str(m)]

    if frame_matches:
        return frame_matches[0]

    return matches[0]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--overlay-folder",
        required=True,
        help="Folder containing selected overlay images with green boxes.",
    )

    parser.add_argument(
        "--output-name",
        default="hard_examples_v2",
        help="Name of output folder inside LabelGUI/RoboflowDetectionUpload.",
    )

    parser.add_argument(
        "--raw-root",
        action="append",
        default=[],
        help="Optional raw image root folder. Can be used multiple times.",
    )

    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the output folder before copying.",
    )

    args = parser.parse_args()

    overlay_folder = Path(args.overlay_folder)

    if not overlay_folder.exists():
        raise FileNotFoundError(f"Overlay folder not found: {overlay_folder}")

    if args.raw_root:
        raw_roots = [Path(p) for p in args.raw_root]
    else:
        raw_roots = DEFAULT_RAW_ROOTS

    output_dir = DEFAULT_OUTPUT_BASE / args.output_name
    output_images_dir = output_dir / "images"

    if args.clear and output_dir.exists():
        shutil.rmtree(output_dir)

    output_images_dir.mkdir(parents=True, exist_ok=True)

    overlay_images = collect_images(overlay_folder)

    if not overlay_images:
        raise FileNotFoundError(f"No images found in overlay folder: {overlay_folder}")

    by_name, by_stem, total_raw_images = build_raw_index(raw_roots)

    copied_rows = []
    missing_rows = []
    duplicate_rows = []

    print("\nCollecting raw hard examples")
    print("============================")
    print(f"Overlay folder: {overlay_folder}")
    print(f"Overlay images selected: {len(overlay_images)}")
    print(f"Raw images indexed: {total_raw_images}")
    print(f"Output folder: {output_images_dir}\n")

    copied_count = 0

    for overlay_path in overlay_images:
        overlay_name = overlay_path.name
        overlay_stem = normalize_stem(overlay_path.stem)

        matches = []

        # First try exact filename.
        if overlay_name in by_name:
            matches = by_name[overlay_name]

        # Then try normalized stem.
        if not matches and overlay_stem in by_stem:
            matches = by_stem[overlay_stem]

        raw_path = choose_best_match(matches, overlay_name)

        if raw_path is None:
            missing_rows.append({
                "overlay_image": str(overlay_path),
                "overlay_filename": overlay_name,
                "reason": "No matching raw image found",
            })
            print(f"[MISSING] {overlay_name}")
            continue

        if len(matches) > 1:
            duplicate_rows.append({
                "overlay_image": str(overlay_path),
                "overlay_filename": overlay_name,
                "chosen_raw": str(raw_path),
                "all_matches": " | ".join(str(m) for m in matches),
            })

        dst = output_images_dir / raw_path.name

        # If same filename already exists, keep it unique.
        if dst.exists():
            dst = output_images_dir / f"{raw_path.stem}__copy{raw_path.suffix}"

        shutil.copy2(raw_path, dst)

        copied_rows.append({
            "overlay_image": str(overlay_path),
            "raw_image": str(raw_path),
            "copied_to": str(dst),
        })

        copied_count += 1
        print(f"[COPIED] {overlay_name}")

    copied_csv = output_dir / "copied_raw_images.csv"
    missing_csv = output_dir / "missing_raw_images.csv"
    duplicates_csv = output_dir / "duplicate_raw_matches.csv"

    with open(copied_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["overlay_image", "raw_image", "copied_to"],
        )
        writer.writeheader()
        writer.writerows(copied_rows)

    with open(missing_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["overlay_image", "overlay_filename", "reason"],
        )
        writer.writeheader()
        writer.writerows(missing_rows)

    with open(duplicates_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["overlay_image", "overlay_filename", "chosen_raw", "all_matches"],
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)

    print("\nDone.")
    print(f"Raw images copied: {copied_count}")
    print(f"Missing matches: {len(missing_rows)}")
    print(f"Duplicate filename matches: {len(duplicate_rows)}")
    print(f"\nUpload this folder to Roboflow:")
    print(output_images_dir)
    print(f"\nLogs:")
    print(copied_csv)
    print(missing_csv)
    print(duplicates_csv)


if __name__ == "__main__":
    main()
