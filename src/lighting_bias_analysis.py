import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config import RESULTS_DIR


def summarize_by_cluster(df, label_col, name):
    df = df.copy()

    df["lighting_gap"] = df["raw_L_mean"] - df["seg_L_mean"]
    df["context_L_noise"] = df["raw_L_std"] - df["seg_L_std"]
    df["context_C_noise"] = df["raw_C_std"] - df["seg_C_std"]
    df["ita_gap"] = df["raw_ITA_mean"] - df["seg_ITA_mean"]

    summary = df.groupby(label_col)[
        [
            "raw_L_mean",
            "seg_L_mean",
            "lighting_gap",
            "context_L_noise",
            "context_C_noise",
            "raw_ITA_mean",
            "seg_ITA_mean",
            "ita_gap",
            "mask_coverage",
        ]
    ].agg(["mean", "std"])

    output_path = RESULTS_DIR / f"{name}_lighting_bias_summary.csv"
    summary.to_csv(output_path)

    print(f"\n=== {name.upper()} LIGHTING BIAS SUMMARY ===")
    print(summary)
    print(f"\nSaved: {output_path}")


def main():
    seg_df = pd.read_csv(RESULTS_DIR / "pseudolabels_k6_balanced_ita_seg.csv")
    raw_df = pd.read_csv(RESULTS_DIR / "pseudolabels_k6_balanced_ita_raw.csv")

    summarize_by_cluster(seg_df, "pseudo_label", "seg_balanced_ita")
    summarize_by_cluster(raw_df, "pseudo_label", "raw_balanced_ita")


if __name__ == "__main__":
    main()