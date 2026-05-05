import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2lab
from tqdm import tqdm

from config import RESULTS_DIR


def load_rgb(path):
    img = Image.open(path).convert("RGB")
    return np.asarray(img).astype(np.float32) / 255.0


def load_mask(path, target_size):
    mask = Image.open(path).convert("L")
    mask = mask.resize(target_size, resample=Image.Resampling.NEAREST)
    mask = np.asarray(mask)
    return mask > 127


def lab_stats(rgb, mask=None):
    lab = rgb2lab(rgb)

    if mask is not None:
        pixels = lab[mask]
    else:
        pixels = lab.reshape(-1, 3)

    if len(pixels) < 50:
        return None

    L = pixels[:, 0]
    a = pixels[:, 1]
    b = pixels[:, 2]

    C = np.sqrt(a**2 + b**2)
    h = np.degrees(np.arctan2(b, a))
    ITA = np.degrees(np.arctan2(L - 50, b + 1e-8))

    return {
        "L_mean": np.mean(L),
        "a_mean": np.mean(a),
        "b_mean": np.mean(b),
        "C_mean": np.mean(C),
        "h_mean": np.mean(h),
        "ITA_mean": np.mean(ITA),
        "L_std": np.std(L),
        "a_std": np.std(a),
        "b_std": np.std(b),
        "C_std": np.std(C),
        "ITA_std": np.std(ITA),
    }


def extract_features():
    metadata_path = RESULTS_DIR / "metadata.csv"
    df = pd.read_csv(metadata_path)

    rows = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        rgb = load_rgb(row["image_path"])
        height, width = rgb.shape[:2]
        mask = load_mask(row["mask_path"], target_size=(width, height)) 

        raw = lab_stats(rgb, mask=None)
        segmented = lab_stats(rgb, mask=mask)

        if raw is None or segmented is None:
            continue

        output = {
            "image_id": row["image_id"],
            "split": row["split"],
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "mask_coverage": float(mask.mean()),
        }

        for k, v in raw.items():
            output[f"raw_{k}"] = v

        for k, v in segmented.items():
            output[f"seg_{k}"] = v

        rows.append(output)

    out_df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "color_features.csv"
    out_df.to_csv(output_path, index=False)

    print(out_df.head())
    print()
    print(out_df["split"].value_counts())
    print()
    print("Mask coverage summary:")
    print(out_df["mask_coverage"].describe())
    print()
    print(f"Saved features to: {output_path}")


if __name__ == "__main__":
    extract_features()