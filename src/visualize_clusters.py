import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from config import RESULTS_DIR


def make_contact_sheet(df, label_col, output_dir, image_type="raw", samples_per_cluster=25):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label in sorted(df[label_col].unique()):
        cluster_df = df[df[label_col] == label].sample(
            n=min(samples_per_cluster, len(df[df[label_col] == label])),
            random_state=42,
        )

        thumbs = []

        for _, row in cluster_df.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")

            if image_type == "seg":
                mask = Image.open(row["mask_path"]).convert("L").resize(img.size)
                bg = Image.new("RGB", img.size, (255, 255, 255))
                img = Image.composite(img, bg, mask)

            img = img.resize((128, 128))
            thumbs.append(img)

        cols = 5
        rows = (len(thumbs) + cols - 1) // cols

        sheet = Image.new("RGB", (cols * 128, rows * 128 + 40), "white")
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 10), f"{image_type.upper()} Cluster {label}", fill="black")

        for i, img in enumerate(thumbs):
            x = (i % cols) * 128
            y = (i // cols) * 128 + 40
            sheet.paste(img, (x, y))

        output_path = output_dir / f"{image_type}_cluster_{label}.jpg"
        sheet.save(output_path)

        print(f"Saved {output_path}")


def main():
    seg_df = pd.read_csv(RESULTS_DIR / "pseudolabels_k6_balanced_ita_seg.csv")
    raw_df = pd.read_csv(RESULTS_DIR / "pseudolabels_k6_balanced_ita_raw.csv")

    make_contact_sheet(
        seg_df,
        label_col="pseudo_label",
        output_dir=RESULTS_DIR / "visual_clusters" / "segmented",
        image_type="seg",
    )

    make_contact_sheet(
        raw_df,
        label_col="pseudo_label",
        output_dir=RESULTS_DIR / "visual_clusters" / "raw",
        image_type="raw",
    )


if __name__ == "__main__":
    main()