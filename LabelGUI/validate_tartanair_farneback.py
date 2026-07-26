import argparse
import io
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from azure.storage.blob import ContainerClient


ACCOUNT_URL = "https://tartanair.blob.core.windows.net/"
CONTAINER_NAME = "tartanair-release1"


def make_client():
    return ContainerClient(
        account_url=ACCOUNT_URL,
        container_name=CONTAINER_NAME,
        credential=None,
    )


def download_blob_bytes(client, blob_name):
    blob_client = client.get_blob_client(blob=blob_name)
    data = blob_client.download_blob()
    return data.content_as_bytes()


def read_image_bgr(client, blob_name):
    raw = download_blob_bytes(client, blob_name)
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not decode image: {blob_name}")

    return img


def read_numpy_blob(client, blob_name):
    raw = download_blob_bytes(client, blob_name)
    return np.load(io.BytesIO(raw))


def list_flow_files(client, traj_dir):
    prefix = f"{traj_dir.rstrip('/')}/flow/"
    files = []

    for blob in client.list_blobs(name_starts_with=prefix):
        name = blob.name
        if name.endswith("_flow.npy"):
            files.append(name)

    return sorted(files)


def parse_flow_filename(flow_blob):
    filename = Path(flow_blob).name

    # Example: 000000_000001_flow.npy
    base = filename.replace("_flow.npy", "")
    parts = base.split("_")

    if len(parts) < 2:
        raise ValueError(f"Unexpected flow filename format: {filename}")

    frame_a = parts[0]
    frame_b = parts[1]

    return frame_a, frame_b


def matching_image_paths(traj_dir, frame_a, frame_b, camera="left"):
    traj_dir = traj_dir.rstrip("/")

    image_a = f"{traj_dir}/image_{camera}/{frame_a}_{camera}.png"
    image_b = f"{traj_dir}/image_{camera}/{frame_b}_{camera}.png"

    return image_a, image_b


def matching_mask_path(flow_blob):
    return flow_blob.replace("_flow.npy", "_mask.npy")


def normalize_gt_flow(flow):
    flow = np.asarray(flow)

    if flow.ndim == 3 and flow.shape[2] >= 2:
        return flow[:, :, :2].astype(np.float32)

    if flow.ndim == 3 and flow.shape[0] == 2:
        return np.transpose(flow[:2, :, :], (1, 2, 0)).astype(np.float32)

    raise ValueError(f"Unexpected ground-truth flow shape: {flow.shape}")


def make_valid_mask(gt_flow, mask=None):
    valid = np.isfinite(gt_flow[:, :, 0]) & np.isfinite(gt_flow[:, :, 1])

    if mask is not None:
        mask = np.asarray(mask)

        if mask.ndim == 3:
            mask = np.squeeze(mask)

        # TartanAir masks may vary by version/source.
        # We choose the polarity that gives a reasonable amount of valid pixels.
        mask_positive = mask > 0
        mask_negative = mask == 0

        if mask_positive.mean() >= 0.05:
            valid = valid & mask_positive
        else:
            valid = valid & mask_negative

    return valid


def compute_farneback_flow(img_a_bgr, img_b_bgr):
    gray_a = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        gray_a,
        gray_b,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0,
    )

    return flow.astype(np.float32)


def flow_to_bgr(flow, max_mag=None):
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    finite_mag = mag[np.isfinite(mag)]

    if max_mag is None:
        if finite_mag.size == 0:
            max_mag = 1.0
        else:
            max_mag = np.percentile(finite_mag, 95)

    max_mag = max(float(max_mag), 1e-6)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = np.clip(ang * 180 / np.pi / 2, 0, 179).astype(np.uint8)
    hsv[:, :, 1] = 255
    hsv[:, :, 2] = np.clip((mag / max_mag) * 255, 0, 255).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def error_to_bgr(error_map, valid_mask):
    error = error_map.copy()
    error[~valid_mask] = 0

    valid_error = error[valid_mask]

    if valid_error.size == 0:
        max_err = 1.0
    else:
        max_err = np.percentile(valid_error, 95)

    max_err = max(float(max_err), 1e-6)

    error_vis = np.clip((error / max_err) * 255, 0, 255).astype(np.uint8)
    error_color = cv2.applyColorMap(error_vis, cv2.COLORMAP_JET)

    invalid_color = np.zeros_like(error_color)
    invalid_color[:, :, :] = (40, 40, 40)

    error_color[~valid_mask] = invalid_color[~valid_mask]

    return error_color


def add_title(img, title):
    out = img.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_visual_example(img_a, img_b, pred_flow, gt_flow, error_map, valid_mask, out_path):
    max_mag = np.percentile(
        np.sqrt(gt_flow[:, :, 0] ** 2 + gt_flow[:, :, 1] ** 2)[valid_mask],
        95,
    ) if valid_mask.any() else None

    pred_vis = flow_to_bgr(pred_flow, max_mag=max_mag)
    gt_vis = flow_to_bgr(gt_flow, max_mag=max_mag)
    err_vis = error_to_bgr(error_map, valid_mask)

    panels = [
        add_title(img_a, "Frame A"),
        add_title(img_b, "Frame B"),
        add_title(pred_vis, "Farneback Estimated Flow"),
        add_title(gt_vis, "TartanAir Ground Truth Flow"),
        add_title(err_vis, "Endpoint Error Map"),
    ]

    # Resize all panels to same display size.
    display_w = 360
    resized = []

    for panel in panels:
        h, w = panel.shape[:2]
        scale = display_w / w
        display_h = int(h * scale)
        resized.append(cv2.resize(panel, (display_w, display_h)))

    row1 = np.hstack(resized[:2])
    row2 = np.hstack(resized[2:4])

    # Error panel alone, padded to same width as rows.
    err = resized[4]
    pad_width = row1.shape[1] - err.shape[1]

    if pad_width > 0:
        pad = np.zeros((err.shape[0], pad_width, 3), dtype=np.uint8)
        err = np.hstack([err, pad])

    combined = np.vstack([row1, row2, err])

    cv2.imwrite(str(out_path), combined)


def evaluate_pair(client, traj_dir, flow_blob, camera, output_dir, visual_index=None):
    frame_a, frame_b = parse_flow_filename(flow_blob)
    image_a_blob, image_b_blob = matching_image_paths(traj_dir, frame_a, frame_b, camera=camera)

    img_a = read_image_bgr(client, image_a_blob)
    img_b = read_image_bgr(client, image_b_blob)

    gt_flow_raw = read_numpy_blob(client, flow_blob)
    gt_flow = normalize_gt_flow(gt_flow_raw)

    mask_blob = matching_mask_path(flow_blob)
    mask = None

    try:
        mask = read_numpy_blob(client, mask_blob)
    except Exception:
        mask = None

    pred_flow = compute_farneback_flow(img_a, img_b)

    if pred_flow.shape[:2] != gt_flow.shape[:2]:
        raise RuntimeError(
            f"Shape mismatch for {flow_blob}: predicted {pred_flow.shape}, ground truth {gt_flow.shape}"
        )

    valid_mask = make_valid_mask(gt_flow, mask=mask)

    diff = pred_flow - gt_flow
    epe_map = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)

    valid_epe = epe_map[valid_mask]

    if valid_epe.size == 0:
        mean_epe = None
        median_epe = None
        p90_epe = None
    else:
        mean_epe = float(np.mean(valid_epe))
        median_epe = float(np.median(valid_epe))
        p90_epe = float(np.percentile(valid_epe, 90))

    if visual_index is not None:
        visual_dir = output_dir / "visual_examples"
        visual_dir.mkdir(parents=True, exist_ok=True)

        out_path = visual_dir / f"pair_{visual_index:03d}_{frame_a}_{frame_b}.jpg"
        make_visual_example(
            img_a,
            img_b,
            pred_flow,
            gt_flow,
            epe_map,
            valid_mask,
            out_path,
        )

    return {
        "flow_blob": flow_blob,
        "image_a": image_a_blob,
        "image_b": image_b_blob,
        "frame_a": frame_a,
        "frame_b": frame_b,
        "valid_pixel_ratio": float(valid_mask.mean()),
        "mean_epe": mean_epe,
        "median_epe": median_epe,
        "p90_epe": p90_epe,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj-dir", default="abandonedfactory/Easy/P001")
    parser.add_argument("--camera", default="left", choices=["left", "right"])
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--visual-examples", type=int, default=8)
    parser.add_argument("--run-name", default="farneback_abandonedfactory_easy_p001")

    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_dir = base_dir / "TartanAirValidation" / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Connecting to TartanAir Azure container...")
    client = make_client()

    print(f"Listing flow files under: {args.traj_dir}/flow/")
    flow_files = list_flow_files(client, args.traj_dir)

    if not flow_files:
        raise RuntimeError(
            f"No flow files found under {args.traj_dir}/flow/. "
            "Try another trajectory, such as abandonedfactory/Easy/P000."
        )

    print(f"Found {len(flow_files)} flow files.")

    selected_flows = flow_files[:: max(args.stride, 1)]
    selected_flows = selected_flows[: args.max_pairs]

    print(f"Testing {len(selected_flows)} pairs.")
    print("This downloads only the selected image/flow pairs, not the full dataset.")

    rows = []

    for i, flow_blob in enumerate(selected_flows):
        print(f"[{i + 1}/{len(selected_flows)}] {flow_blob}")

        visual_index = i if i < args.visual_examples else None

        try:
            row = evaluate_pair(
                client=client,
                traj_dir=args.traj_dir,
                flow_blob=flow_blob,
                camera=args.camera,
                output_dir=output_dir,
                visual_index=visual_index,
            )
            rows.append(row)

        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({
                "flow_blob": flow_blob,
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    per_pair_csv = output_dir / "per_pair_errors.csv"
    df.to_csv(per_pair_csv, index=False)

    valid_df = df.dropna(subset=["mean_epe"]) if "mean_epe" in df.columns else pd.DataFrame()

    metrics = {
        "method": "OpenCV Farneback dense optical flow",
        "opencv_function": "cv2.calcOpticalFlowFarneback",
        "tartanair_traj_dir": args.traj_dir,
        "camera": args.camera,
        "pairs_requested": args.max_pairs,
        "pairs_evaluated": int(len(valid_df)),
        "mean_epe_over_pairs": float(valid_df["mean_epe"].mean()) if len(valid_df) else None,
        "median_epe_over_pairs": float(valid_df["median_epe"].median()) if len(valid_df) else None,
        "mean_p90_epe_over_pairs": float(valid_df["p90_epe"].mean()) if len(valid_df) else None,
        "mean_valid_pixel_ratio": float(valid_df["valid_pixel_ratio"].mean()) if len(valid_df) else None,
        "output_dir": str(output_dir),
        "per_pair_csv": str(per_pair_csv),
    }

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print(f"Metrics: {metrics_path}")
    print(f"Per-pair errors: {per_pair_csv}")
    print(f"Visual examples: {output_dir / 'visual_examples'}")
    print("\nSummary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
