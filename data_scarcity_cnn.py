import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class FacePseudoLabelDataset(Dataset):
    def __init__(self, df, image_col, label_col, image_size=128):
        self.df = df.reset_index(drop=True)
        self.image_col = image_col
        self.label_col = label_col
        self.image_size = image_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image = cv2.imread(str(row[self.image_col]))

        if image is None:
            raise FileNotFoundError(row[self.image_col])

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        label = int(row[self.label_col])

        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)

        return x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def infer_columns(df):
    image_candidates = ["image_path", "path", "img_path", "raw_image_path"]
    label_candidates = ["label", "pseudo_label", "cluster", "cluster_id", "ita_bin", "balanced_ita_label"]

    image_col = next((c for c in image_candidates if c in df.columns), None)
    label_col = next((c for c in label_candidates if c in df.columns), None)

    if image_col is None:
        raise ValueError(f"Could not infer image column. Columns are: {list(df.columns)}")

    if label_col is None:
        raise ValueError(f"Could not infer label column. Columns are: {list(df.columns)}")

    return image_col, label_col


def load_pseudolabel_data(pseudolabel_csv, metadata_csv=None):
    labels = pd.read_csv(pseudolabel_csv)

    if "split" in labels.columns and any(col in labels.columns for col in ["image_path", "path", "img_path", "raw_image_path"]):
        return labels

    if metadata_csv is None:
        raise ValueError("Need metadata_csv because pseudolabel file does not contain both split and image paths.")

    metadata = pd.read_csv(metadata_csv)

    join_col = None

    for candidate in ["image_id", "id", "filename"]:
        if candidate in labels.columns and candidate in metadata.columns:
            join_col = candidate
            break

    if join_col is None:
        raise ValueError("Could not find shared join column between pseudolabels and metadata.")

    merged = metadata.merge(labels, on=join_col, how="inner")

    return merged


def subsample_by_class(train_df, label_col, keep_fraction, seed):
    sampled_parts = []

    for label, group in train_df.groupby(label_col):
        n_keep = max(1, int(round(len(group) * keep_fraction)))
        sampled = group.sample(n=n_keep, random_state=seed)
        sampled_parts.append(sampled)

    return pd.concat(sampled_parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_one_model(train_df, eval_df, image_col, label_col, num_classes, device, args):
    train_dataset = FacePseudoLabelDataset(train_df, image_col, label_col, args.image_size)
    eval_dataset = FacePseudoLabelDataset(eval_df, image_col, label_col, args.image_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in eval_loader:
            images = images.to(device)

            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def plot_results(results_df, output_path):
    plt.figure()
    plt.plot(results_df["percent_kept"], results_df["accuracy"], marker="o", label="Accuracy")
    plt.plot(results_df["percent_kept"], results_df["balanced_accuracy"], marker="o", label="Balanced Accuracy")
    plt.plot(results_df["percent_kept"], results_df["macro_f1"], marker="o", label="Macro F1")
    plt.xlabel("Training data kept (%)")
    plt.ylabel("Score")
    plt.title("CNN Performance Under Per-Cluster Training Data Scarcity")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pseudolabel_csv", required=True)
    parser.add_argument("--metadata_csv", default=None)
    parser.add_argument("--output_dir", default="results/data_scarcity_cnn")
    parser.add_argument("--eval_split", default="test")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_pseudolabel_data(args.pseudolabel_csv, args.metadata_csv)

    image_col, label_col = infer_columns(df)

    labels_sorted = sorted(df[label_col].unique())
    label_map = {old_label: new_label for new_label, old_label in enumerate(labels_sorted)}
    df[label_col] = df[label_col].map(label_map)

    num_classes = df[label_col].nunique()

    train_df = df[df["split"] == "train"].copy()
    eval_df = df[df["split"] == args.eval_split].copy()

    keep_fractions = [1.00, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Image column: {image_col}")
    print(f"Label column: {label_col}")
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_df)}")
    print(f"Eval samples: {len(eval_df)}")

    all_results = []

    for keep_fraction in keep_fractions:
        percent_kept = int(keep_fraction * 100)

        print(f"\nTraining with {percent_kept}% of each pseudo-label cluster kept")

        sub_train_df = subsample_by_class(train_df, label_col, keep_fraction, args.seed)

        metrics = train_one_model(
            sub_train_df,
            eval_df,
            image_col,
            label_col,
            num_classes,
            device,
            args,
        )

        row = {
            "percent_kept": percent_kept,
            "keep_fraction": keep_fraction,
            "train_samples": len(sub_train_df),
            **metrics,
        }

        all_results.append(row)

        print(row)

    results_df = pd.DataFrame(all_results)

    csv_path = output_dir / "cnn_data_scarcity_results.csv"
    plot_path = output_dir / "cnn_data_scarcity_curve.png"

    results_df.to_csv(csv_path, index=False)
    plot_results(results_df, plot_path)

    print(f"\nSaved results to: {csv_path}")
    print(f"Saved plot to: {plot_path}")


if __name__ == "__main__":
    main()