import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, pearsonr

from config import RESULTS_DIR


MSKCC_DIR = Path("data/mskcc")
OUT_DIR = RESULTS_DIR / "mskcc"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def compute_ita(L, b):
    return np.degrees(np.arctan2(L - 50, b + 1e-8))


def correlation_report(df, x_col, y_col, name):
    clean = df[[x_col, y_col]].dropna()

    pearson = pearsonr(clean[x_col], clean[y_col])
    spearman = spearmanr(clean[x_col], clean[y_col])

    return {
        "comparison": name,
        "n": len(clean),
        "pearson_r": pearson.statistic,
        "pearson_p": pearson.pvalue,
        "spearman_r": spearman.statistic,
        "spearman_p": spearman.pvalue,
    }


def cluster_agreement(df, feature_cols, label_col, name, k=6):
    clean = df[feature_cols + [label_col]].dropna().copy()

    X = clean[feature_cols].values
    y = clean[label_col].astype(str).values

    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    clusters = kmeans.fit_predict(X_scaled)

    return {
        "comparison": name,
        "n": len(clean),
        "k": k,
        "ari": adjusted_rand_score(y, clusters),
        "nmi": normalized_mutual_info_score(y, clusters),
        "silhouette": silhouette_score(X_scaled, clusters),
        "davies_bouldin": davies_bouldin_score(X_scaled, clusters),
        "calinski_harabasz": calinski_harabasz_score(X_scaled, clusters),
    }


def main():
    s5 = pd.read_csv(MSKCC_DIR / "s5.csv")
    s7 = pd.read_csv(MSKCC_DIR / "s7.csv")
    metadata = pd.read_csv(MSKCC_DIR / "mskcc-skin-tone-labeling-dataset.csv")

    print("\nLoaded:")
    print("s5:", s5.shape)
    print("s7:", s7.shape)
    print("metadata:", metadata.shape)

    # s5 has colorimeter Lab values + expert labels.
    s5 = s5.copy()
    s5["computed_ita"] = compute_ita(s5["average_l"], s5["average_b"])

    correlation_rows = []

    correlation_rows.append(
        correlation_report(
            s5,
            "computed_ita",
            "fst",
            "colorimeter ITA vs Fitzpatrick Skin Type",
        )
    )

    correlation_rows.append(
        correlation_report(
            s5,
            "computed_ita",
            "mst_r1",
            "colorimeter ITA vs Monk Skin Tone Rater 1",
        )
    )

    correlation_rows.append(
        correlation_report(
            s5,
            "computed_ita",
            "mst_r2",
            "colorimeter ITA vs Monk Skin Tone Rater 2",
        )
    )

    # s7 has image-extracted Lab/ITA and colorimeter average Lab/ITA.
    correlation_rows.append(
        correlation_report(
            s7,
            "img_ita",
            "average_ita",
            "image-extracted ITA vs colorimeter ITA",
        )
    )

    correlation_df = pd.DataFrame(correlation_rows)
    correlation_path = OUT_DIR / "mskcc_correlations.csv"
    correlation_df.to_csv(correlation_path, index=False)

    print("\n=== CORRELATIONS ===")
    print(correlation_df)

    agreement_rows = []

    # Cluster colorimeter Lab values and compare to known labels.
    lab_features = ["average_l", "average_a", "average_b"]

    agreement_rows.append(
        cluster_agreement(
            s5,
            lab_features,
            "fst",
            "KMeans Lab clusters vs FST",
            k=6,
        )
    )

    agreement_rows.append(
        cluster_agreement(
            s5,
            lab_features,
            "mst_r1",
            "KMeans Lab clusters vs MST Rater 1",
            k=6,
        )
    )

    agreement_rows.append(
        cluster_agreement(
            s5,
            lab_features,
            "mst_r2",
            "KMeans Lab clusters vs MST Rater 2",
            k=6,
        )
    )

    # Cluster image-extracted Lab from s7 and compare against colorimeter ITA bins.
    s7 = s7.copy()
    s7["average_ita_bin_6"] = pd.qcut(
        s7["average_ita"],
        q=6,
        labels=False,
        duplicates="drop",
    )

    image_lab_features = ["img_l", "img_a", "img_b"]

    agreement_rows.append(
        cluster_agreement(
            s7,
            image_lab_features,
            "average_ita_bin_6",
            "KMeans image Lab clusters vs colorimeter ITA quantile bins",
            k=6,
        )
    )

    agreement_df = pd.DataFrame(agreement_rows)
    agreement_path = OUT_DIR / "mskcc_cluster_agreement.csv"
    agreement_df.to_csv(agreement_path, index=False)

    print("\n=== CLUSTER AGREEMENT ===")
    print(agreement_df)

    # Simple label counts for report.
    counts = {
        "s5_fst_counts": s5["fst"].value_counts(dropna=False).sort_index(),
        "s5_mst_r1_counts": s5["mst_r1"].value_counts(dropna=False).sort_index(),
        "s5_mst_r2_counts": s5["mst_r2"].value_counts(dropna=False).sort_index(),
        "metadata_fst_counts": metadata["fitzpatrick_skin_type"].value_counts(dropna=False).sort_index(),
    }

    for name, series in counts.items():
        path = OUT_DIR / f"{name}.csv"
        series.to_csv(path)
        print(f"\n{name}:")
        print(series)

    print("\nSaved outputs to:")
    print(correlation_path)
    print(agreement_path)
    print(OUT_DIR)


if __name__ == "__main__":
    main()