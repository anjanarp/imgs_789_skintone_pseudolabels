import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
from sklearn.cluster import KMeans
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


def main(k=6):
    df = pd.read_csv(RESULTS_DIR / "color_features.csv")

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    scaler = StandardScaler()

    X_train = scaler.fit_transform(train_df[SEG_FEATURES])
    X_val = scaler.transform(val_df[SEG_FEATURES])
    X_test = scaler.transform(test_df[SEG_FEATURES])

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)

    train_df["pseudo_label"] = kmeans.fit_predict(X_train)
    val_df["pseudo_label"] = kmeans.predict(X_val)
    test_df["pseudo_label"] = kmeans.predict(X_test)

    labeled_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    output_csv = RESULTS_DIR / "pseudolabels_k6.csv"
    labeled_df.to_csv(output_csv, index=False)

    joblib.dump(scaler, RESULTS_DIR / "k6_scaler.joblib")
    joblib.dump(kmeans, RESULTS_DIR / "k6_kmeans.joblib")

    print("\nPseudo-label counts by split:")
    print(pd.crosstab(labeled_df["split"], labeled_df["pseudo_label"]))

    print("\nOverall pseudo-label counts:")
    print(labeled_df["pseudo_label"].value_counts().sort_index())

    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main(k=6)