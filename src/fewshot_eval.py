import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
)

from config import RESULTS_DIR


SEG_FEATURES = [
    "seg_L_mean",
    "seg_a_mean",
    "seg_b_mean",
    "seg_C_mean",
    "seg_h_mean",
    "seg_ITA_mean",
]

RAW_FEATURES = [
    "raw_L_mean",
    "raw_a_mean",
    "raw_b_mean",
    "raw_C_mean",
    "raw_h_mean",
    "raw_ITA_mean",
]


def sample_support_set(df, k):
    support = []
    for cls in sorted(df["pseudo_label"].unique()):
        cls_df = df[df["pseudo_label"] == cls]

        # safety in case a class has < k samples
        k_eff = min(k, len(cls_df))

        support.append(cls_df.sample(n=k_eff, random_state=42))
    return pd.concat(support)


def compute_prototypes(support_df, features):
    prototypes = {}
    for cls in support_df["pseudo_label"].unique():
        cls_feats = support_df[support_df["pseudo_label"] == cls][features].values
        prototypes[cls] = cls_feats.mean(axis=0)
    return prototypes


def predict_soft(query_feats, prototypes):
    probs = []
    preds = []

    for x in query_feats:
        distances = np.array([
            np.linalg.norm(x - prototypes[cls])
            for cls in sorted(prototypes.keys())
        ])

        # convert distances → similarity
        sims = np.exp(-distances)

        # normalize to probabilities
        p = sims / sims.sum()

        probs.append(p)
        preds.append(np.argmax(p))

    return np.array(preds), np.array(probs)

def top_k_accuracy(probs, y_true, k):
    top_k = np.argsort(probs, axis=1)[:, -k:]
    correct = [y in top_k[i] for i, y in enumerate(y_true)]
    return np.mean(correct)


def run_fewshot(file, features, k):
    df = pd.read_csv(file)

    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    support = sample_support_set(train_df, k)
    prototypes = compute_prototypes(support, features)

    X_test = test_df[features].values
    y_test = test_df["pseudo_label"].values

    preds, probs = predict_soft(X_test, prototypes)

    return {
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_test, preds),
        "confusion": confusion_matrix(y_test, preds),
        "top2": top_k_accuracy(probs, y_test, 2),
        "top3": top_k_accuracy(probs, y_test, 3),
    }


def print_results(name, metrics):
    print(f"{name}:")
    print(f"  accuracy:       {metrics['accuracy']:.4f}")
    print(f"  macro_f1:       {metrics['macro_f1']:.4f}")
    print(f"  balanced_acc:   {metrics['balanced_acc']:.4f}")
    print(f"  top2_acc:       {metrics['top2']:.4f}")
    print(f"  top3_acc:       {metrics['top3']:.4f}")


if __name__ == "__main__":
    seg_file = RESULTS_DIR / "pseudolabels_k6.csv"
    raw_file = RESULTS_DIR / "pseudolabels_k6_raw.csv"

    for k in [1, 3, 5]:
        print(f"\n====================")
        print(f"{k}-SHOT RESULTS")
        print(f"====================")

        seg_metrics = run_fewshot(seg_file, SEG_FEATURES, k)
        raw_metrics = run_fewshot(raw_file, RAW_FEATURES, k)

        print_results("SEGMENTED", seg_metrics)
        print_results("RAW", raw_metrics)

        print("\nConfusion Matrix (SEGMENTED):")
        print(seg_metrics["confusion"])

        print("\nConfusion Matrix (RAW):")
        print(raw_metrics["confusion"])