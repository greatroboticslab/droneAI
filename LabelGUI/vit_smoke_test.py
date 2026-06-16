import argparse
import json
import random
from pathlib import Path
from datetime import datetime

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

        return image, label_idx, str(row["image_path"])


# ------------------------------------------------------------
# Load manifest
# ------------------------------------------------------------
def load_manifest():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {MANIFEST_PATH}. Run Frame Extraction first."
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

    # A clip group must include the session name because many sessions may have
    # clip names like 001_takeoff.mp4.
    df["clip_group"] = df["session_name"] + "__" + df["clip_filename"]

    return df


# ------------------------------------------------------------
# Split helpers
# ------------------------------------------------------------
def make_train_val_split(df, split_mode="image", test_size=0.25):
    """
    image:
        Random frame-level split.
        Fast smoke test, but can overestimate performance.

    clip:
        Keeps all frames from the same clip together.

    session:
        Keeps all frames from the same labeled video/session together.
        This is the most honest, but needs more data.
    """

    split_note = ""

    if split_mode == "image":
        label_counts = df["label"].value_counts().to_dict()
        min_class_count = min(label_counts.values())
        stratify_col = df["label"] if min_class_count >= 2 else None

        train_df, val_df = train_test_split(
            df,
            test_size=test_size,
            random_state=42,
            stratify=stratify_col,
        )

        split_note = (
            "Image-level split. This is useful for a smoke test, but it can "
            "overestimate performance because frames from the same clip may "
            "appear in both training and validation."
        )

        return train_df, val_df, split_note

    if split_mode == "clip":
        group_col = "clip_group"
        split_note = (
            "Clip-level split. Frames from the same clip are kept together. "
            "This is more honest than image-level splitting."
        )

    elif split_mode == "session":
        group_col = "session_name"
        split_note = (
            "Session-level split. Frames from the same labeled video/session "
            "are kept together. This is the most honest split, but it needs "
            "more labeled videos to be stable."
        )

    else:
        raise ValueError(f"Unknown split mode: {split_mode}")

    unique_groups = df[group_col].nunique()

    if unique_groups < 2:
        print("[WARNING] Not enough groups for group split. Falling back to image split.")
        return make_train_val_split(df, split_mode="image", test_size=test_size)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=42,
    )

    train_idx, val_idx = next(
        splitter.split(df, groups=df[group_col])
    )

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    if train_df.empty or val_df.empty:
        print("[WARNING] Group split created an empty train or validation set. Falling back to image split.")
        return make_train_val_split(df, split_mode="image", test_size=test_size)

    train_labels = set(train_df["label"].unique())
    val_labels = set(val_df["label"].unique())

    missing_train = sorted(set(df["label"].unique()) - train_labels)
    missing_val = sorted(set(df["label"].unique()) - val_labels)

    if missing_train:
        print("[WARNING] These labels are missing from training split:", missing_train)

    if missing_val:
        print("[WARNING] These labels are missing from validation split:", missing_val)

    return train_df, val_df, split_note


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------
def build_model(num_classes, use_pretrained=True):
    weights = None

    if use_pretrained:
        try:
            weights = ViT_B_16_Weights.DEFAULT
            model = vit_b_16(weights=weights)
            print("[INFO] Loaded pretrained ViT weights.")
        except Exception as e:
            print("[WARNING] Could not load pretrained weights.")
            print("[WARNING] Reason:", e)
            print("[WARNING] Continuing with random weights. Results will not be meaningful.")
            model = vit_b_16(weights=None)
    else:
        model = vit_b_16(weights=None)

    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)

    # Freeze backbone for quick test
    for param in model.parameters():
        param.requires_grad = False

    # Train only classification head
    for param in model.heads.parameters():
        param.requires_grad = True

    if weights is not None:
        transform = weights.transforms()
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return model, transform


# ------------------------------------------------------------
# Training / Evaluation
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    for images, labels, _paths in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(labels.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    return avg_loss, acc


def evaluate(model, loader, criterion, device, idx_to_label):
    model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_paths = []
    all_confidences = []

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            confidences, preds = probs.max(dim=1)

            total_loss += loss.item() * images.size(0)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(labels.detach().cpu().numpy().tolist())
            all_confidences.extend(confidences.detach().cpu().numpy().tolist())
            all_paths.extend(paths)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)

    rows = []

    for path, true_idx, pred_idx, confidence in zip(
        all_paths, all_labels, all_preds, all_confidences
    ):
        rows.append({
            "image_path": path,
            "true_label": idx_to_label[true_idx],
            "predicted_label": idx_to_label[pred_idx],
            "confidence": round(float(confidence), 4),
            "correct": bool(true_idx == pred_idx),
        })

    return avg_loss, acc, all_labels, all_preds, rows


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="image",
        choices=["image", "clip", "session"],
        help="image = random frame split, clip = clip-level split, session = video/session-level split",
    )

    args = parser.parse_args()

    set_seed(42)

    print("\n==============================")
    print("DroneAI ViT Test")
    print("==============================\n")

    print(f"[INFO] Split mode: {args.split_mode}")

    df = load_manifest()

    label_counts = df["label"].value_counts().to_dict()
    labels = sorted(df["label"].unique().tolist())

    if len(labels) < 2:
        raise ValueError("Need at least 2 labels for training.")

    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    print("\n[INFO] Labels found:")
    for label, count in label_counts.items():
        print(f"  - {label}: {count} images")

    print(f"\n[INFO] Sessions found: {df['session_name'].nunique()}")
    print(f"[INFO] Clips found: {df['clip_group'].nunique()}")

    train_df, val_df, split_note = make_train_val_split(
        df,
        split_mode=args.split_mode,
        test_size=0.25,
    )

    print(f"\n[INFO] Training images: {len(train_df)}")
    print(f"[INFO] Validation images: {len(val_df)}")

    print("\n[INFO] Training label counts:")
    print(train_df["label"].value_counts().to_string())

    print("\n[INFO] Validation label counts:")
    print(val_df["label"].value_counts().to_string())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[INFO] Device: {device}")

    model, transform = build_model(
        num_classes=len(labels),
        use_pretrained=not args.no_pretrained,
    )

    model = model.to(device)

    train_dataset = FrameDataset(train_df, label_to_idx, transform=transform)
    val_dataset = FrameDataset(val_df, label_to_idx, transform=transform)

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

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )

    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_true, val_pred, pred_rows = evaluate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_label,
        )

        row = {
            "epoch": epoch,
            "train_loss": round(float(train_loss), 4),
            "train_acc": round(float(train_acc), 4),
            "val_loss": round(float(val_loss), 4),
            "val_acc": round(float(val_acc), 4),
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    val_loss, val_acc, val_true, val_pred, pred_rows = evaluate(
        model,
        val_loader,
        criterion,
        device,
        idx_to_label,
    )

    class_names = [idx_to_label[i] for i in range(len(labels))]

    report = classification_report(
        val_true,
        val_pred,
        labels=list(range(len(labels))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        val_true,
        val_pred,
        labels=list(range(len(labels))),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"vit_{args.split_mode}_test_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "vit_model.pth"

    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "split_mode": args.split_mode,
    }, model_path)

    with open(run_dir / "labels.json", "w", encoding="utf-8") as f:
        json.dump({
            "labels": labels,
            "label_to_idx": label_to_idx,
            "idx_to_label": idx_to_label,
        }, f, indent=2)

    metrics = {
        "created_at": datetime.now().isoformat(),
        "device": str(device),
        "split_mode": args.split_mode,
        "total_images": int(len(df)),
        "train_images": int(len(train_df)),
        "val_images": int(len(val_df)),
        "total_sessions": int(df["session_name"].nunique()),
        "total_clips": int(df["clip_group"].nunique()),
        "train_sessions": int(train_df["session_name"].nunique()),
        "val_sessions": int(val_df["session_name"].nunique()),
        "train_clips": int(train_df["clip_group"].nunique()),
        "val_clips": int(val_df["clip_group"].nunique()),
        "label_counts": label_counts,
        "train_label_counts": train_df["label"].value_counts().to_dict(),
        "val_label_counts": val_df["label"].value_counts().to_dict(),
        "final_val_loss": round(float(val_loss), 4),
        "final_val_acc": round(float(val_acc), 4),
        "history": history,
        "classification_report": report,
        "note": split_note,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    pd.DataFrame(pred_rows).to_csv(run_dir / "prediction_samples.csv", index=False)

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(run_dir / "confusion_matrix.csv")

    # Save train/val split for inspection
    train_df.to_csv(run_dir / "train_split.csv", index=False)
    val_df.to_csv(run_dir / "val_split.csv", index=False)

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Results saved to: {run_dir}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print("\nImportant: Compare image, clip, and session splits. Session split is the most honest, but needs more data.\n")


if __name__ == "__main__":
    main()
