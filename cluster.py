import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler

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

def evaluate_clustering(X, labels, name):
    return {
        "feature_set": name,
        "silhouette": silhouette_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
    }
    
def run_clustering(df, features, name, k):
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[features])
    X_val = scaler.transform(val_df[features])
    X_test = scaler.transform(test_df[features])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)

    train_labels = kmeans.fit_predict(X_train)
    val_labels = kmeans.predict(X_val)
    test_labels = kmeans.predict(X_test)

    metrics = [
        evaluate_clustering(X_train, train_labels, f"{name}_train"),
        evaluate_clustering(X_val, val_labels, f"{name}_val"),
        evaluate_clustering(X_test, test_labels, f"{name}_test"),
    ]

    return pd.DataFrame(metrics)


def main():
    df = pd.read_csv(RESULTS_DIR / "color_features.csv")

    all_results = []

    for k in [3, 6, 8, 10]:
        print(f"\n=== K = {k} ===")

        seg_metrics = run_clustering(df, SEG_FEATURES, f"seg_k{k}", k)
        raw_metrics = run_clustering(df, RAW_FEATURES, f"raw_k{k}", k)

        combined = pd.concat([seg_metrics, raw_metrics], ignore_index=True)
        combined["k"] = k

        print(combined)

        all_results.append(combined)

    final_df = pd.concat(all_results, ignore_index=True)

    output_path = RESULTS_DIR / "raw_vs_segmented_k_sweep_metrics.csv"
    final_df.to_csv(output_path, index=False)

    print("\n=== FINAL RAW VS SEGMENTED K SWEEP ===")
    print(final_df)
    print(f"\nSaved metrics to: {output_path}")
    
if __name__ == "__main__":
    main()