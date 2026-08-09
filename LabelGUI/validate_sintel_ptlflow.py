import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

import ptlflow
from ptlflow.utils.io_adapter import IOAdapter


FLO_TAG = 202021.25


def read_flo(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        tag = np.fromfile(f, np.float32, count=1)[0]

        if float(tag) != FLO_TAG:
            raise ValueError(f"Invalid .flo tag in {path}. Got {tag}")

        width = int(np.fromfile(f, np.int32, count=1)[0])
        height = int(np.fromfile(f, np.int32, count=1)[0])

        data = np.fromfile(f, np.float32, count=2 * width * height)

    if data.size != 2 * width * height:
        raise ValueError(f"Unexpected .flo size in {path}")

    return data.reshape((height, width, 2)).astype(np.float32)


def find_modality_dir(root: Path, modality_name: str) -> Path:
    candidates = []

    for p in root.rglob(modality_name):
        if not p.is_dir():
            continue

        child_dirs = [c for c in p.iterdir() if c.is_dir()]
        if child_dirs:
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"Could not find Sintel modality folder '{modality_name}' under {root}"
        )

    for p in candidates:
        if p.parent.name.lower() == "training":
            return p

    return candidates[0]


def list_flow_pairs(root: Path, pass_name: str):
    image_dir = find_modality_dir(root, pass_name)
    flow_dir = find_modality_dir(root, "flow")

    invalid_dir = None
    try:
        invalid_dir = find_modality_dir(root, "invalid")
    except Exception:
        invalid_dir = None

    pairs = []

    for flow_path in sorted(flow_dir.glob("*/*.flo")):
        sequence = flow_path.parent.name
        frame_name = flow_path.stem

        try:
            frame_num = int(frame_name.replace("frame_", ""))
        except Exception:
            continue

        img_a = image_dir / sequence / f"frame_{frame_num:04d}.png"
        img_b = image_dir / sequence / f"frame_{frame_num + 1:04d}.png"

        if not img_a.exists() or not img_b.exists():
            continue

        invalid_path = None
        if invalid_dir is not None:
            candidate = invalid_dir / sequence / f"frame_{frame_num:04d}.png"
            if candidate.exists():
                invalid_path = candidate

        pairs.append({
            "sequence": sequence,
            "frame": frame_name,
            "img_a": img_a,
            "img_b": img_b,
            "flow": flow_path,
            "invalid": invalid_path,
        })

    return pairs


def read_image(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    return img


def read_invalid_mask(path: Path, shape_hw):
    if path is None or not path.exists():
        return np.ones(shape_hw, dtype=bool)

    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        return np.ones(shape_hw, dtype=bool)

    if mask.shape != shape_hw:
        mask = cv2.resize(mask, (shape_hw[1], shape_hw[0]), interpolation=cv2.INTER_NEAREST)

    # Sintel invalid masks: nonzero usually marks invalid pixels.
    return mask == 0


def move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)

    if isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)

    return obj


def resize_flow_to_gt(pred_flow: np.ndarray, gt_shape_hw):
    target_h, target_w = gt_shape_hw
    pred_h, pred_w = pred_flow.shape[:2]

    if pred_h == target_h and pred_w == target_w:
        return pred_flow

    resized = cv2.resize(pred_flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    scale_x = target_w / pred_w
    scale_y = target_h / pred_h

    resized[:, :, 0] *= scale_x
    resized[:, :, 1] *= scale_y

    return resized.astype(np.float32)


def compute_ptlflow_flow(img_a_bgr, img_b_bgr, model, device):
    images = [img_a_bgr, img_b_bgr]

    # IOAdapter handles input formatting, padding, scaling, and tensor conversion.
    try:
        io_adapter = IOAdapter(model, images[0].shape[:2], cuda=(device.type == "cuda"))
    except TypeError:
        io_adapter = IOAdapter(model, images[0].shape[:2])

    inputs = io_adapter.prepare_inputs(images)
    inputs = move_to_device(inputs, device)

    with torch.inference_mode():
        predictions = model(inputs)

    # Some PTLFlow versions expose an output postprocess helper.
    if hasattr(io_adapter, "unpad_and_unscale"):
        try:
            predictions = io_adapter.unpad_and_unscale(predictions)
        except Exception:
            pass

    if "flows" not in predictions:
        raise RuntimeError(f"Model output did not contain 'flows'. Keys: {list(predictions.keys())}")

    flows = predictions["flows"]

    # Expected shape is usually B x N x C x H x W, e.g. 1 x 1 x 2 x H x W.
    if flows.ndim == 5:
        flow = flows[0, 0]
    elif flows.ndim == 4:
        flow = flows[0]
    else:
        raise RuntimeError(f"Unexpected flows shape: {flows.shape}")

    flow = flow.detach().cpu().float().numpy()
    flow = np.transpose(flow, (1, 2, 0))

    return flow.astype(np.float32)


def endpoint_error(pred: np.ndarray, gt: np.ndarray, valid: np.ndarray):
    diff = pred - gt
    epe_map = np.sqrt(diff[:, :, 0] ** 2 + diff[:, :, 1] ** 2)

    finite = np.isfinite(epe_map) & np.isfinite(gt[:, :, 0]) & np.isfinite(gt[:, :, 1])
    valid = valid & finite

    values = epe_map[valid]

    if values.size == 0:
        return {
            "mean_epe": None,
            "median_epe": None,
            "p90_epe": None,
            "valid_pixel_ratio": float(valid.mean()),
            "epe_map": epe_map,
            "valid_mask": valid,
        }

    return {
        "mean_epe": float(values.mean()),
        "median_epe": float(np.median(values)),
        "p90_epe": float(np.percentile(values, 90)),
        "valid_pixel_ratio": float(valid.mean()),
        "epe_map": epe_map,
        "valid_mask": valid,
    }


def flow_to_bgr(flow: np.ndarray, max_mag=None) -> np.ndarray:
    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=False)

    if max_mag is None:
        finite_mag = mag[np.isfinite(mag)]
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


def error_to_bgr(epe_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    error = epe_map.copy()
    error[~valid_mask] = 0

    valid_values = error[valid_mask]

    if valid_values.size == 0:
        max_error = 1.0
    else:
        max_error = np.percentile(valid_values, 95)

    max_error = max(float(max_error), 1e-6)

    gray = np.clip((error / max_error) * 255, 0, 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    color[~valid_mask] = (40, 40, 40)

    return color


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()

    cv2.rectangle(out, (0, 0), (out.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(
        out,
        title,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def resize_panel(img: np.ndarray, width: int = 420) -> np.ndarray:
    h, w = img.shape[:2]
    scale = width / w
    new_h = int(h * scale)
    return cv2.resize(img, (width, new_h))


def save_visual_example(img_a, img_b, pred_flow, gt_flow, epe_map, valid_mask, out_path: Path, model_label: str):
    gt_mag = np.sqrt(gt_flow[:, :, 0] ** 2 + gt_flow[:, :, 1] ** 2)

    if valid_mask.any():
        max_mag = np.percentile(gt_mag[valid_mask], 95)
    else:
        max_mag = None

    pred_vis = flow_to_bgr(pred_flow, max_mag=max_mag)
    gt_vis = flow_to_bgr(gt_flow, max_mag=max_mag)
    err_vis = error_to_bgr(epe_map, valid_mask)

    panels = [
        add_title(img_a, "Frame A"),
        add_title(img_b, "Frame B"),
        add_title(pred_vis, f"{model_label} Estimated Flow"),
        add_title(gt_vis, "Sintel Ground Truth Flow"),
        add_title(err_vis, "Endpoint Error Map"),
    ]

    panels = [resize_panel(p, width=420) for p in panels]

    row1 = np.hstack([panels[0], panels[1]])
    row2 = np.hstack([panels[2], panels[3]])

    err = panels[4]
    pad_width = row1.shape[1] - err.shape[1]

    if pad_width > 0:
        pad = np.zeros((err.shape[0], pad_width, 3), dtype=np.uint8)
        err = np.hstack([err, pad])

    combined = np.vstack([row1, row2, err])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), combined)


def load_model(model_name: str, ckpt_path: str, device):
    print(f"Loading model: {model_name}")
    print(f"Checkpoint: {ckpt_path}")

    model = ptlflow.get_model(model_name, ckpt_path=ckpt_path)
    model = model.to(device)
    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sintel-root", default="LabelGUI/SintelData")
    parser.add_argument("--pass-name", default="clean", choices=["clean", "final"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--max-pairs", type=int, default=5)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--visual-examples", type=int, default=3)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    device = torch.device(args.device)

    base_dir = Path(__file__).resolve().parent
    sintel_root = Path(args.sintel_root)

    if not sintel_root.is_absolute():
        sintel_root = Path.cwd() / sintel_root

    run_name = args.run_name
    if not run_name:
        safe_ckpt = args.ckpt.replace("/", "_").replace("\\", "_").replace(".", "_")
        run_name = f"{args.model}_{safe_ckpt}_{args.pass_name}_{args.max_pairs}"

    output_dir = base_dir / "SintelValidation" / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Sintel root:", sintel_root)
    print("Output dir:", output_dir)
    print("Pass:", args.pass_name)
    print("Model:", args.model)
    print("Checkpoint:", args.ckpt)
    print("Device:", device)

    pairs = list_flow_pairs(sintel_root, args.pass_name)

    if not pairs:
        raise RuntimeError(
            "No Sintel frame/flow pairs found. Check that the training images and training extras were extracted."
        )

    selected = pairs[:: max(args.stride, 1)]
    selected = selected[: args.max_pairs]

    print(f"Found {len(pairs)} possible frame pairs.")
    print(f"Testing {len(selected)} pairs.")

    model = load_model(args.model, args.ckpt, device)

    rows = []

    for i, pair in enumerate(selected):
        print(f"[{i + 1}/{len(selected)}] {pair['sequence']} {pair['frame']}")

        try:
            img_a = read_image(pair["img_a"])
            img_b = read_image(pair["img_b"])
            gt_flow = read_flo(pair["flow"])

            pred_flow = compute_ptlflow_flow(img_a, img_b, model=model, device=device)
            pred_flow = resize_flow_to_gt(pred_flow, gt_flow.shape[:2])

            valid_mask = read_invalid_mask(pair["invalid"], gt_flow.shape[:2])
            err = endpoint_error(pred_flow, gt_flow, valid_mask)

            row = {
                "sequence": pair["sequence"],
                "frame": pair["frame"],
                "img_a": str(pair["img_a"]),
                "img_b": str(pair["img_b"]),
                "flow_gt": str(pair["flow"]),
                "invalid_mask": str(pair["invalid"]) if pair["invalid"] else "",
                "mean_epe": err["mean_epe"],
                "median_epe": err["median_epe"],
                "p90_epe": err["p90_epe"],
                "valid_pixel_ratio": err["valid_pixel_ratio"],
            }

            rows.append(row)

            if i < args.visual_examples:
                out_path = output_dir / "visual_examples" / f"{i:03d}_{pair['sequence']}_{pair['frame']}.jpg"
                save_visual_example(
                    img_a,
                    img_b,
                    pred_flow,
                    gt_flow,
                    err["epe_map"],
                    err["valid_mask"],
                    out_path,
                    model_label=args.model,
                )

        except Exception as e:
            print("  ERROR:", e)
            rows.append({
                "sequence": pair.get("sequence", ""),
                "frame": pair.get("frame", ""),
                "error": str(e),
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "per_pair_errors.csv"
    df.to_csv(csv_path, index=False)

    valid_df = df.dropna(subset=["mean_epe"]) if "mean_epe" in df.columns else pd.DataFrame()

    metrics = {
        "method": f"PTLFlow {args.model}",
        "model": args.model,
        "checkpoint": args.ckpt,
        "dataset": "MPI Sintel training set",
        "pass_name": args.pass_name,
        "pairs_requested": args.max_pairs,
        "pairs_evaluated": int(len(valid_df)),
        "device": str(device),
        "mean_epe_over_pairs": float(valid_df["mean_epe"].mean()) if len(valid_df) else None,
        "median_epe_over_pairs": float(valid_df["median_epe"].median()) if len(valid_df) else None,
        "mean_p90_epe_over_pairs": float(valid_df["p90_epe"].mean()) if len(valid_df) else None,
        "mean_valid_pixel_ratio": float(valid_df["valid_pixel_ratio"].mean()) if len(valid_df) else None,
        "output_dir": str(output_dir),
        "per_pair_csv": str(csv_path),
    }

    metrics_path = output_dir / "metrics.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nDone.")
    print("Metrics:", metrics_path)
    print("Per-pair errors:", csv_path)
    print("Visual examples:", output_dir / "visual_examples")
    print("\nSummary:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
