import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision import transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
FRAME_DATASET_DIR = BASE_DIR / "FrameDataset"
MANIFEST_PATH = FRAME_DATASET_DIR / "frame_manifest.csv"

RESULTS_DIR = BASE_DIR / "ViTResults"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------
class FrameDataset(Dataset):
    def __init__(self, dataframe, label_to_idx, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = FRAME_DATASET_DIR / row["image_path"]
        label_name = row["label"]
        label_idx = self.label_to_idx[label_name]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {
            "image": image,
            "label_idx": label_idx,
            "image_path": str(row["image_path"]),
            "label_name": label_name,
            "clip_group": row["clip_group"],
            "clip_filename": row["clip_filename"],
            "session_name": row["session_name"],
        }


# ------------------------------------------------------------
# Load manifest
# ------------------------------------------------------------
def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MANIFEST_PATH}. Run frame extraction first."
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_cols = {"image_path", "label", "session_name", "clip_filename"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Manifest is missing columns: {missing}")

    existing_rows = []

    for _, row in df.iterrows():
        image_path = FRAME_DATASET_DIR / row["image_path"]
        if image_path.exists():
            existing_rows.append(row)

    df = pd.DataFrame(existing_rows)

    if df.empty:
        raise ValueError("No valid images found in FrameDataset.")

    df["label"] = df["label"].astype(str)
    df["session_name"] = df["session_name"].astype(str)
    df["clip_filename"] = df["clip_filename"].astype(str)

    # One clip must be unique by source/session + clip filename.
    df["clip_group"] = df["session_name"] + "__" + df["clip_filename"]

    return df


# ------------------------------------------------------------
# Split helpers
# ------------------------------------------------------------
def make_train_val_split(df, split_mode="clip", test_size=0.25):
    """
    image:
        Random frame-level split.
        Useful for smoke tests, but can overestimate accuracy.

    clip:
        Keeps frames from the same clip together.
        This is the main metric for your professor's 3-second clip test.

    session:
        Keeps full labeled sessions/videos together.
        Most honest, but needs more data.
    """

    labels = sorted(df["label"].unique().tolist())

    if split_mode == "image":
        label_counts = df["label"].value_counts()
        stratify_col = df["label"] if label_counts.min() >= 2 else None

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_col,
        )

        note = (
            "Image-level split. This can overestimate performance because "
            "similar frames from the same clip may appear in both train and validation."
        )

        return train_df.copy(), val_df.copy(), note

    if split_mode == "clip":
        clip_df = (
            df[["clip_group", "label"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        label_counts = clip_df["label"].value_counts()
        stratify_col = clip_df["label"] if label_counts.min() >= 2 else None

        train_clips, val_clips = train_test_split(
            clip_df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_col,
        )

        train_groups = set(train_clips["clip_group"].tolist())
        val_groups = set(val_clips["clip_group"].tolist())

        train_df = df[df["clip_group"].isin(train_groups)].copy()
        val_df = df[df["clip_group"].isin(val_groups)].copy()

        note = (
            "Clip-level split. Frames from the same clip are kept together. "
            "This is the main metric for testing 3-second clip classification."
        )

        return train_df, val_df, note

    if split_mode == "session":
        if df["session_name"].nunique() < 2:
            print("[WARNING] Not enough sessions for session split. Falling back to clip split.")
            return make_train_val_split(df, split_mode="clip", test_size=test_size)

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=42,
        )

        train_idx, val_idx = next(
            splitter.split(df, groups=df["session_name"])
        )

        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()

        note = (
            "Session-level split. Full labeled videos/sessions are kept together. "
            "This is the most honest split, but it needs more labeled sessions."
        )

        return train_df, val_df, note

    raise ValueError(f"Unknown split mode: {split_mode}")


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_vit_model(num_classes, unfreeze_last_n=0):
    """
    Uses pretrained ViT-B/16.

    Default:
        Freeze backbone and train only classification head.

    If unfreeze_last_n > 0:
        Also fine-tune the last N transformer encoder layers.
        This is slower and better suited for GPU/cluster.
    """

    weights = ViT_B_16_Weights.DEFAULT
    model = vit_b_16(weights=weights)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Train classification head
    for param in model.heads.parameters():
        param.requires_grad = True

    # Optional fine-tuning of last transformer blocks
    if unfreeze_last_n > 0:
        try:
            layers = list(model.encoder.layers.children())
            for layer in layers[-unfreeze_last_n:]:
                for param in layer.parameters():
                    param.requires_grad = True

            if hasattr(model.encoder, "ln"):
                for param in model.encoder.ln.parameters():
                    param.requires_grad = True

            print(f"[INFO] Unfroze last {unfreeze_last_n} ViT encoder layers.")
        except Exception as e:
            print("[WARNING] Could not unfreeze encoder layers:", e)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.03,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = weights.transforms()

    return model, train_transform, val_transform


# ------------------------------------------------------------
# Training / Evaluation
# ------------------------------------------------------------
def batch_to_device(batch, device):
    images = batch["image"].to(device)
    labels = batch["label_idx"].to(device)
    return images, labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in loader:
        images, labels = batch_to_device(batch, device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)

        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    frame_acc = accuracy_score(all_true, all_pred)

    return avg_loss, frame_acc


def evaluate_frame_and_clip(model, loader, criterion, device, idx_to_label):
    model.eval()

    total_loss = 0.0

    frame_true = []
    frame_pred = []
    frame_rows = []

    clip_probs = defaultdict(list)
    clip_true = {}
    clip_meta = {}

    with torch.no_grad():
        for batch in loader:
            images, labels = batch_to_device(batch, device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            total_loss += loss.item() * images.size(0)

            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            conf_np = confidences.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

            for i in range(len(labels_np)):
                true_idx = int(labels_np[i])
                pred_idx = int(preds_np[i])

                clip_group = batch["clip_group"][i]
                image_path = batch["image_path"][i]
                clip_filename = batch["clip_filename"][i]
                session_name = batch["session_name"][i]

                frame_true.append(true_idx)
                frame_pred.append(pred_idx)

                frame_rows.append({
                    "image_path": image_path,
                    "session_name": session_name,
                    "clip_group": clip_group,
                    "clip_filename": clip_filename,
                    "true_label": idx_to_label[true_idx],
                    "predicted_label": idx_to_label[pred_idx],
                    "confidence": round(float(conf_np[i]), 4),
                    "correct": bool(true_idx == pred_idx),
                })

                clip_probs[clip_group].append(probs_np[i])
                clip_true[clip_group] = true_idx
                clip_meta[clip_group] = {
                    "session_name": session_name,
                    "clip_filename": clip_filename,
                }

    avg_loss = total_loss / len(loader.dataset)
    frame_acc = accuracy_score(frame_true, frame_pred)

    clip_true_list = []
    clip_pred_list = []
    clip_rows = []

    for clip_group, probs_list in clip_probs.items():
        mean_probs = np.mean(np.stack(probs_list, axis=0), axis=0)
        pred_idx = int(np.argmax(mean_probs))
        confidence = float(np.max(mean_probs))
        true_idx = int(clip_true[clip_group])

        clip_true_list.append(true_idx)
        clip_pred_list.append(pred_idx)

        clip_rows.append({
            "clip_group": clip_group,
            "session_name": clip_meta[clip_group]["session_name"],
            "clip_filename": clip_meta[clip_group]["clip_filename"],
            "true_label": idx_to_label[true_idx],
            "predicted_label": idx_to_label[pred_idx],
            "confidence": round(confidence, 4),
            "num_frames": len(probs_list),
            "correct": bool(true_idx == pred_idx),
        })

    clip_acc = accuracy_score(clip_true_list, clip_pred_list)

    return {
        "loss": avg_loss,
        "frame_acc": frame_acc,
        "clip_acc": clip_acc,
        "frame_true": frame_true,
        "frame_pred": frame_pred,
        "clip_true": clip_true_list,
        "clip_pred": clip_pred_list,
        "frame_rows": frame_rows,
        "clip_rows": clip_rows,
    }


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--split-mode",
        type=str,
        default="clip",
        choices=["image", "clip", "session"],
    )
    parser.add_argument(
        "--unfreeze-last-n",
        type=int,
        default=0,
        help="0 = train only classifier head. 1 or 2 = fine-tune last ViT blocks. Use GPU/cluster for this.",
    )
    args = parser.parse_args()

    set_seed(42)

    print("\n==============================")
    print("DroneAI ViT Clip Classifier")
    print("==============================\n")

    print(f"[INFO] Split mode: {args.split_mode}")
    print(f"[INFO] Epochs: {args.epochs}")
    print(f"[INFO] Batch size: {args.batch_size}")
    print(f"[INFO] Learning rate: {args.lr}")
    print(f"[INFO] Unfreeze last N layers: {args.unfreeze_last_n}")

    df = load_manifest()

    labels = sorted(df["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    total_clip_counts = (
        df[["clip_group", "label"]]
        .drop_duplicates()
        ["label"]
        .value_counts()
        .to_dict()
    )

    total_frame_counts = df["label"].value_counts().to_dict()

    print("\n[INFO] Clip counts by label:")
    for label, count in total_clip_counts.items():
        print(f"  - {label}: {count} clips")

    print("\n[INFO] Frame counts by label:")
    for label, count in total_frame_counts.items():
        print(f"  - {label}: {count} frames")

    if len(labels) < 2:
        raise ValueError("Need at least 2 labels.")

    train_df, val_df, split_note = make_train_val_split(
        df,
        split_mode=args.split_mode,
        test_size=0.25,
    )

    train_clip_counts = (
        train_df[["clip_group", "label"]]
        .drop_duplicates()
        ["label"]
        .value_counts()
        .to_dict()
    )

    val_clip_counts = (
        val_df[["clip_group", "label"]]
        .drop_duplicates()
        ["label"]
        .value_counts()
        .to_dict()
    )

    print("\n[INFO] Training clips:", train_df["clip_group"].nunique())
    print("[INFO] Validation clips:", val_df["clip_group"].nunique())

    print("\n[INFO] Training clip counts by label:")
    for label, count in train_clip_counts.items():
        print(f"  - {label}: {count}")

    print("\n[INFO] Validation clip counts by label:")
    for label, count in val_clip_counts.items():
        print(f"  - {label}: {count}")

    missing_train = sorted(set(labels) - set(train_df["label"].unique()))
    missing_val = sorted(set(labels) - set(val_df["label"].unique()))

    if missing_train:
        print("[WARNING] Labels missing from training:", missing_train)

    if missing_val:
        print("[WARNING] Labels missing from validation:", missing_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    model, train_transform, val_transform = build_vit_model(
        num_classes=len(labels),
        unfreeze_last_n=args.unfreeze_last_n,
    )

    model = model.to(device)

    train_dataset = FrameDataset(train_df, label_to_idx, transform=train_transform)
    val_dataset = FrameDataset(val_df, label_to_idx, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    # Class weighting based on training frame counts.
    train_label_counts = train_df["label"].value_counts().to_dict()
    weights = []

    for label in labels:
        count = train_label_counts.get(label, 1)
        weights.append(1.0 / count)

    weights = torch.tensor(weights, dtype=torch.float32)
    weights = weights / weights.sum() * len(weights)
    weights = weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    history = []
    best_clip_acc = -1.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_frame_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_eval = evaluate_frame_and_clip(
            model,
            val_loader,
            criterion,
            device,
            idx_to_label,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_frame_acc), 4),
            "val_loss": round(float(val_eval["loss"]), 4),
            "val_acc": round(float(val_eval["clip_acc"]), 4),
            "val_frame_acc": round(float(val_eval["frame_acc"]), 4),
            "val_clip_acc": round(float(val_eval["clip_acc"]), 4),
        }

        history.append(row)

        if val_eval["clip_acc"] > best_clip_acc:
            best_clip_acc = val_eval["clip_acc"]
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, "
            f"train_frame_acc={train_frame_acc:.4f} | "
            f"val_loss={val_eval['loss']:.4f}, "
            f"val_frame_acc={val_eval['frame_acc']:.4f}, "
            f"val_clip_acc={val_eval['clip_acc']:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    final_eval = evaluate_frame_and_clip(
        model,
        val_loader,
        criterion,
        device,
        idx_to_label,
    )

    class_names = [idx_to_label[i] for i in range(len(labels))]

    frame_report = classification_report(
        final_eval["frame_true"],
        final_eval["frame_pred"],
        labels=list(range(len(labels))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    clip_report = classification_report(
        final_eval["clip_true"],
        final_eval["clip_pred"],
        labels=list(range(len(labels))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    frame_cm = confusion_matrix(
        final_eval["frame_true"],
        final_eval["frame_pred"],
        labels=list(range(len(labels))),
    )

    clip_cm = confusion_matrix(
        final_eval["clip_true"],
        final_eval["clip_pred"],
        labels=list(range(len(labels))),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"vit_clip_classifier_{args.split_mode}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "vit_clip_classifier.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "split_mode": args.split_mode,
        "unfreeze_last_n": args.unfreeze_last_n,
        "model_name": "vit_b_16",
    }, model_path)

    with open(run_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
        }, f, indent=2)

    metrics = {
        "created_at": datetime.now().isoformat(),
        "model_name": "vit_b_16",
        "device": str(device),
        "split_mode": args.split_mode,
        "main_metric": "clip_accuracy",
        "total_images": int(len(df)),
        "total_clips": int(df["clip_group"].nunique()),
        "total_sessions": int(df["session_name"].nunique()),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "train_clips": int(train_df["clip_group"].nunique()),
        "val_clips": int(val_df["clip_group"].nunique()),
        "train_sessions": int(train_df["session_name"].nunique()),
        "val_sessions": int(val_df["session_name"].nunique()),
        "label_counts": total_frame_counts,
        "clip_label_counts": total_clip_counts,
        "train_clip_label_counts": train_clip_counts,
        "val_clip_label_counts": val_clip_counts,
        "final_val_acc": round(float(final_eval["clip_acc"]), 4),
        "final_clip_acc": round(float(final_eval["clip_acc"]), 4),
        "final_frame_acc": round(float(final_eval["frame_acc"]), 4),
        "final_val_loss": round(float(final_eval["loss"]), 4),
        "best_clip_acc": round(float(best_clip_acc), 4),
        "history": history,
        "classification_report": clip_report,
        "frame_classification_report": frame_report,
        "note": split_note + " Main reported accuracy is clip-level accuracy.",
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(final_eval["frame_rows"]).to_csv(
        run_dir / "prediction_samples.csv",
        index=False,
    )

    pd.DataFrame(final_eval["clip_rows"]).to_csv(
        run_dir / "clip_predictions.csv",
        index=False,
    )

    pd.DataFrame(clip_cm, index=class_names, columns=class_names).to_csv(
        run_dir / "confusion_matrix.csv"
    )

    pd.DataFrame(frame_cm, index=class_names, columns=class_names).to_csv(
        run_dir / "frame_confusion_matrix.csv"
    )

    train_df.to_csv(run_dir / "train_split.csv", index=False)
    val_df.to_csv(run_dir / "val_split.csv", index=False)

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Results saved to: {run_dir}")
    print(f"Final clip accuracy: {final_eval['clip_acc']:.4f}")
    print(f"Final frame accuracy: {final_eval['frame_acc']:.4f}")
    print("\nMain number to report right now: clip accuracy.\n")


if __name__ == "__main__":
    main()
