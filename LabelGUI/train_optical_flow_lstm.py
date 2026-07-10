import argparse
import json
import random
from pathlib import Path
from datetime import datetime
from collections import Counter

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent

DEFAULT_SEQUENCE_CSV = (
    BASE_DIR / "OpticalFlowResults" / "flow_v1" / "flow_sequence_features.csv"
)

RESULTS_DIR = BASE_DIR / "OpticalFlowResults"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_numeric_bool(series):
    return series.astype(str).str.lower().isin(["true", "1", "yes", "y"]).astype(float)


def load_sequence_data(sequence_csv: Path):
    if not sequence_csv.exists():
        raise FileNotFoundError(f"Could not find: {sequence_csv}")

    df = pd.read_csv(sequence_csv)

    required = {
        "clip_group",
        "session_name",
        "clip_filename",
        "label",
        "step_index",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["clip_group"] = df["clip_group"].astype(str)
    df["session_name"] = df["session_name"].astype(str)
    df["clip_filename"] = df["clip_filename"].astype(str)
    df["label"] = df["label"].astype(str)
    df["step_index"] = pd.to_numeric(df["step_index"], errors="coerce").fillna(0)

    # Convert boolean-like columns to numeric 0/1.
    for col in ["roi_available", "detected_a", "detected_b", "both_detected", "any_detected"]:
        if col in df.columns:
            df[col] = to_numeric_bool(df[col])

    # These are the sequence features we want the LSTM to see.
    preferred_features = [
        "roi_available",
        "both_detected",
        "any_detected",

        "flow_dx_norm_per_sec",
        "flow_dy_norm_per_sec",
        "flow_mag_norm_per_sec",

        "flow_dx_mean_per_sec",
        "flow_dy_mean_per_sec",
        "flow_mag_mean_per_sec",

        "flow_mag_mean",
        "flow_mag_median",
        "flow_mag_max",
        "flow_mag_std",

        "det_vx_norm_per_sec",
        "det_vy_norm_per_sec",
        "det_speed_norm_per_sec",

        "det_dx",
        "det_dy",
        "det_speed",

        "conf_a",
        "conf_b",

        "roi_width",
        "roi_height",
    ]

    feature_cols = [c for c in preferred_features if c in df.columns]

    if not feature_cols:
        raise ValueError("No optical-flow feature columns found.")

    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    clip_rows = []

    for clip_group, group in df.groupby("clip_group"):
        group = group.copy()
        group = group.sort_values("step_index")

        label = group["label"].iloc[0]
        session_name = group["session_name"].iloc[0]
        clip_filename = group["clip_filename"].iloc[0]

        sequence = group[feature_cols].to_numpy(dtype=np.float32)

        clip_rows.append({
            "clip_group": clip_group,
            "session_name": session_name,
            "clip_filename": clip_filename,
            "label": label,
            "sequence": sequence,
            "length": len(sequence),
        })

    clips_df = pd.DataFrame(clip_rows)

    return clips_df, feature_cols


def make_split(clips_df, split_mode="clip", test_size=0.25):
    labels = clips_df["label"].astype(str)
    indices = np.arange(len(clips_df))

    if split_mode == "clip":
        label_counts = labels.value_counts()
        stratify = labels if label_counts.min() >= 2 else None

        train_idx, val_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=stratify,
        )

        note = "Clip-level split. Random held-out clips are used for validation."
        return train_idx, val_idx, note

    if split_mode == "session":
        groups = clips_df["session_name"].astype(str)

        all_labels = set(labels.unique())

        best_train = None
        best_val = None

        for seed in range(42, 200):
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=seed,
            )

            train_idx, val_idx = next(splitter.split(clips_df, labels, groups=groups))

            train_labels = set(labels.iloc[train_idx].unique())
            val_labels = set(labels.iloc[val_idx].unique())

            if all_labels.issubset(train_labels) and all_labels.issubset(val_labels):
                best_train = train_idx
                best_val = val_idx
                break

        if best_train is None:
            splitter = GroupShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=42,
            )
            best_train, best_val = next(splitter.split(clips_df, labels, groups=groups))

        note = "Session-level split. Full source sessions/videos are held out."
        return best_train, best_val, note

    raise ValueError(f"Unknown split mode: {split_mode}")


def compute_standardizer(clips_df, train_idx):
    all_train_steps = []

    for idx in train_idx:
        all_train_steps.append(clips_df.iloc[idx]["sequence"])

    all_train_steps = np.concatenate(all_train_steps, axis=0)

    mean = all_train_steps.mean(axis=0)
    std = all_train_steps.std(axis=0)

    std[std < 1e-6] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


class FlowSequenceDataset(Dataset):
    def __init__(self, clips_df, indices, label_to_idx, mean, std):
        self.clips_df = clips_df.reset_index(drop=True)
        self.indices = list(indices)
        self.label_to_idx = label_to_idx
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        row = self.clips_df.iloc[idx]

        seq = row["sequence"].astype(np.float32)
        seq = (seq - self.mean) / self.std

        label_name = row["label"]
        label_idx = self.label_to_idx[label_name]

        return {
            "sequence": torch.tensor(seq, dtype=torch.float32),
            "length": int(row["length"]),
            "label_idx": int(label_idx),
            "label_name": label_name,
            "clip_group": row["clip_group"],
            "session_name": row["session_name"],
            "clip_filename": row["clip_filename"],
        }


def collate_batch(batch):
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label_idx"] for item in batch], dtype=torch.long)

    max_len = int(lengths.max().item())
    feature_dim = batch[0]["sequence"].shape[1]

    padded = torch.zeros((len(batch), max_len, feature_dim), dtype=torch.float32)

    for i, item in enumerate(batch):
        seq = item["sequence"]
        padded[i, : seq.shape[0], :] = seq

    return {
        "sequence": padded,
        "lengths": lengths,
        "labels": labels,
        "label_names": [item["label_name"] for item in batch],
        "clip_groups": [item["clip_group"] for item in batch],
        "session_names": [item["session_name"] for item in batch],
        "clip_filenames": [item["clip_filename"] for item in batch],
    }


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes, dropout=0.25, bidirectional=True):
        super().__init__()

        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

    def forward(self, x, lengths):
        lengths_cpu = lengths.detach().cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths_cpu,
            batch_first=True,
            enforce_sorted=False,
        )

        _, (h_n, _) = self.lstm(packed)

        if self.bidirectional:
            # Last layer forward and backward hidden states.
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        logits = self.classifier(h)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in loader:
        x = batch["sequence"].to(device)
        lengths = batch["lengths"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(x, lengths)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)

        all_true.extend(labels.detach().cpu().numpy().tolist())
        all_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc


def evaluate(model, loader, criterion, device, idx_to_label):
    model.eval()

    total_loss = 0.0
    all_true = []
    all_pred = []
    rows = []

    with torch.no_grad():
        for batch in loader:
            x = batch["sequence"].to(device)
            lengths = batch["lengths"].to(device)
            labels = batch["labels"].to(device)

            logits = model(x, lengths)
            loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)

            total_loss += loss.item() * x.size(0)

            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            conf_np = conf.detach().cpu().numpy()
            probs_np = probs.detach().cpu().numpy()

            for i in range(len(labels_np)):
                true_idx = int(labels_np[i])
                pred_idx = int(preds_np[i])

                row = {
                    "clip_group": batch["clip_groups"][i],
                    "session_name": batch["session_names"][i],
                    "clip_filename": batch["clip_filenames"][i],
                    "true_label": idx_to_label[true_idx],
                    "predicted_label": idx_to_label[pred_idx],
                    "confidence": float(conf_np[i]),
                    "correct": bool(true_idx == pred_idx),
                }

                for class_idx, label_name in idx_to_label.items():
                    row[f"prob_{label_name}"] = float(probs_np[i, class_idx])

                rows.append(row)

            all_true.extend(labels_np.tolist())
            all_pred.extend(preds_np.tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_true, all_pred)

    return avg_loss, acc, all_true, all_pred, rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sequence-csv",
        default=str(DEFAULT_SEQUENCE_CSV),
        help="Path to flow_sequence_features.csv",
    )

    parser.add_argument(
        "--split-mode",
        choices=["clip", "session"],
        default="clip",
    )

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--test-size", type=float, default=0.25)

    args = parser.parse_args()

    set_seed(42)

    sequence_csv = Path(args.sequence_csv)

    clips_df, feature_cols = load_sequence_data(sequence_csv)

    labels = sorted(clips_df["label"].unique().tolist())
    label_to_idx = {label: i for i, label in enumerate(labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}

    train_idx, val_idx, split_note = make_split(
        clips_df,
        split_mode=args.split_mode,
        test_size=args.test_size,
    )

    mean, std = compute_standardizer(clips_df, train_idx)

    train_dataset = FlowSequenceDataset(clips_df, train_idx, label_to_idx, mean, std)
    val_dataset = FlowSequenceDataset(clips_df, val_idx, label_to_idx, mean, std)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_batch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LSTMClassifier(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_classes=len(labels),
        dropout=args.dropout,
        bidirectional=True,
    ).to(device)

    train_labels = [clips_df.iloc[i]["label"] for i in train_idx]
    train_counts = Counter(train_labels)

    class_weights = []

    for label in labels:
        class_weights.append(1.0 / max(1, train_counts[label]))

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    class_weights = class_weights / class_weights.sum() * len(class_weights)
    class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-3,
    )

    best_val_acc = -1.0
    best_state = None
    history = []

    print("\n==============================")
    print("DroneAI Optical Flow LSTM")
    print("==============================\n")

    print(f"[INFO] Sequence CSV: {sequence_csv}")
    print(f"[INFO] Total clips: {len(clips_df)}")
    print(f"[INFO] Feature count: {len(feature_cols)}")
    print(f"[INFO] Labels: {labels}")
    print(f"[INFO] Split mode: {args.split_mode}")
    print(f"[INFO] Device: {device}")

    print("\n[INFO] Clip counts by label:")
    for label, count in clips_df["label"].value_counts().items():
        print(f"  - {label}: {count}")

    print("\n[INFO] Train clips:", len(train_idx))
    print("[INFO] Val clips:", len(val_idx))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, _, _, _ = evaluate(
            model,
            val_loader,
            criterion,
            device,
            idx_to_label,
        )

        history.append({
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f}, "
            f"train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    val_loss, val_acc, y_true, y_pred, pred_rows = evaluate(
        model,
        val_loader,
        criterion,
        device,
        idx_to_label,
    )

    class_names = [idx_to_label[i] for i in range(len(labels))]

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(labels))),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    run_dir = (
        RESULTS_DIR
        / f"flow_lstm_{args.split_mode}_{timestamp}"
    )

    run_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "labels": labels,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "feature_cols": feature_cols,
        "mean": mean,
        "std": std,
        "args": vars(args),
    }, run_dir / "flow_lstm_model.pth")

    pd.DataFrame(pred_rows).to_csv(run_dir / "predictions.csv", index=False)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(run_dir / "confusion_matrix.csv", index=True)

    train_split = clips_df.iloc[train_idx].drop(columns=["sequence"]).copy()
    val_split = clips_df.iloc[val_idx].drop(columns=["sequence"]).copy()

    train_split.to_csv(run_dir / "train_split.csv", index=False)
    val_split.to_csv(run_dir / "val_split.csv", index=False)

    metrics = {
        "created_at": datetime.now().isoformat(),
        "sequence_csv": str(sequence_csv),
        "model": "bidirectional_lstm",
        "note": "This is a standard PyTorch LSTM sequence baseline, not true xLSTM yet.",
        "split_mode": args.split_mode,
        "test_size": args.test_size,
        "total_clips": int(len(clips_df)),
        "train_clips": int(len(train_idx)),
        "val_clips": int(len(val_idx)),
        "labels": labels,
        "label_counts": clips_df["label"].value_counts().to_dict(),
        "final_val_acc": float(val_acc),
        "best_val_acc": float(best_val_acc),
        "final_val_loss": float(val_loss),
        "classification_report": report,
        "history": history,
        "feature_cols": feature_cols,
        "split_note": split_note,
    }

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n==============================")
    print("Finished")
    print("==============================")
    print(f"Results saved to: {run_dir}")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Final validation accuracy: {val_acc:.4f}")
    print("\nMain files:")
    print(f"  {run_dir / 'metrics.json'}")
    print(f"  {run_dir / 'predictions.csv'}")
    print(f"  {run_dir / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
