import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import RESULTS_DIR


def assign_quantile_labels(df, feature, output_name, k=6):
    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()
    test_df = df[df["split"] == "test"].copy()

    train_df["pseudo_label"], bins = pd.qcut(
        train_df[feature],
        q=k,
        labels=False,
        retbins=True,
        duplicates="drop",
    )

    for split_df in [val_df, test_df]:
        split_df["pseudo_label"] = pd.cut(
            split_df[feature],
            bins=bins,
            labels=False,
            include_lowest=True,
        )

        split_df["pseudo_label"] = split_df["pseudo_label"].fillna(0).astype(int)

    out_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    output_path = RESULTS_DIR / output_name
    out_df.to_csv(output_path, index=False)

    print(f"\n=== {output_name} ===")
    print("Feature:", feature)
    print("Bins:")
    print(bins)
    print("\nPseudo-label counts by split:")
    print(pd.crosstab(out_df["split"], out_df["pseudo_label"]))
    print(f"\nSaved: {output_path}")


def main():
    df = pd.read_csv(RESULTS_DIR / "color_features.csv")

    assign_quantile_labels(
        df=df,
        feature="seg_ITA_mean",
        output_name="pseudolabels_k6_balanced_ita_seg.csv",
        k=6,
    )

    assign_quantile_labels(
        df=df,
        feature="raw_ITA_mean",
        output_name="pseudolabels_k6_balanced_ita_raw.csv",
        k=6,
    )


if __name__ == "__main__":
    main()