import argparse
import csv
import os
from pathlib import Path

import cv2
from inference_sdk import InferenceHTTPClient


BASE_DIR = Path(__file__).resolve().parent


def collect_images(source_dir, limit=None):
    source_dir = Path(source_dir)
    exts = {".jpg", ".jpeg", ".png"}

    images = []

    for path in source_dir.rglob("*"):
        if path.suffix.lower() in exts:
            images.append(path)

    images = sorted(images)

    if limit:
        images = images[:limit]

    return images


def draw_predictions(image_path, predictions, output_path):
    image = cv2.imread(str(image_path))

    if image is None:
        return False

    for pred in predictions:
        x = float(pred.get("x", 0))
        y = float(pred.get("y", 0))
        w = float(pred.get("width", 0))
        h = float(pred.get("height", 0))
        confidence = float(pred.get("confidence", 0))
        class_name = pred.get("class", "object")

        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label = f"{class_name} {confidence:.2f}"
        cv2.putText(
            image,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Folder of images to test")
    parser.add_argument("--model-id", required=True, help="Roboflow model id, example: drone-detection/1")
    parser.add_argument("--api-key", default="", help="Roboflow API key. Or set ROBOFLOW_API_KEY env variable.")
    parser.add_argument("--run-name", default="roboflow_detector_overlay")
    parser.add_argument("--limit", type=int, default=100)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY", "")

    if not api_key:
        raise ValueError("Missing API key. Pass --api-key or set ROBOFLOW_API_KEY.")

    source_dir = Path(args.source)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder not found: {source_dir}")

    output_dir = BASE_DIR / "DetectionOverlays" / args.run_name
    images_out_dir = output_dir / "images"
    csv_path = output_dir / "detections.csv"

    images_out_dir.mkdir(parents=True, exist_ok=True)

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )

    images = collect_images(source_dir, limit=args.limit)

    rows = []

    print(f"[INFO] Found {len(images)} images.")
    print(f"[INFO] Saving overlays to: {images_out_dir}")

    for idx, image_path in enumerate(images, start=1):
        print(f"[{idx}/{len(images)}] {image_path.name}")

        result = client.infer(str(image_path), model_id=args.model_id)
        predictions = result.get("predictions", [])

        output_path = images_out_dir / image_path.name
        draw_predictions(image_path, predictions, output_path)

        if predictions:
            for pred in predictions:
                rows.append({
                    "image": str(image_path),
                    "overlay": str(output_path),
                    "class": pred.get("class", ""),
                    "confidence": pred.get("confidence", ""),
                    "x": pred.get("x", ""),
                    "y": pred.get("y", ""),
                    "width": pred.get("width", ""),
                    "height": pred.get("height", ""),
                })
        else:
            rows.append({
                "image": str(image_path),
                "overlay": str(output_path),
                "class": "",
                "confidence": "",
                "x": "",
                "y": "",
                "width": "",
                "height": "",
                "note": "no detection",
            })

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "image",
            "overlay",
            "class",
            "confidence",
            "x",
            "y",
            "width",
            "height",
            "note",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)

    print("\nDone.")
    print(f"Overlay images: {images_out_dir}")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
