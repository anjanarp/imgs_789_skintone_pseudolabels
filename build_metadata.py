import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from config import IMAGE_DIR, MASK_DIR, SPLIT_DIR, RESULTS_DIR


def read_ids(split_path):
    with open(split_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def build_metadata():
    rows = []

    for split in ["train", "val", "test"]:
        ids = read_ids(SPLIT_DIR / f"{split}_ids.txt")

        for image_id in ids:
            image_path = IMAGE_DIR / f"{image_id}.jpg"
            # mask_path = MASK_DIR / split / f"{image_id}.png"
            mask_path = MASK_DIR / f"{image_id}.png"

            if image_path.exists() and mask_path.exists():
                rows.append(
                    {
                        "image_id": image_id,
                        "split": split,
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                    }
                )

    df = pd.DataFrame(rows)
    output_path = RESULTS_DIR / "metadata.csv"
    df.to_csv(output_path, index=False)

    print(df.head())
    print()
    print(df["split"].value_counts())
    print()
    print(f"Saved metadata to: {output_path}")


if __name__ == "__main__":
    build_metadata()