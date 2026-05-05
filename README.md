# Segmentation-Aware Colorimetry for Skin Appearance Pseudo-Labeling

This repository implements a segmentation-aware colorimetry pipeline for studying whether skin appearance pseudo-labels can be learned from image-derived CIELAB/ITA features under limited ground truth. The project began as a downstream extension of my computer vision segmentation project and evolved into an ML4DD-style investigation of pseudo-label reliability, few-shot learnability, lighting bias, and external colorimetry validation.

## Core Research Question

Can raw or segmented image-derived color features produce meaningful skin appearance clusters, and are those pseudo-labels learnable under few-shot constraints?

The final conclusion is nuanced: CIELAB/ITA features are meaningful when measured under controlled conditions, but extracting them from unconstrained images introduces lighting and context noise. The pipeline can learn structure, but on CelebA that structure often reflects global brightness/exposure rather than true skin tone.

## High-Level Project Story

1. I started from my CV project, which produced conservative engineered skin masks for CelebA faces.
2. I extracted CIELAB/ITA features from both raw images and segmented skin-only regions.
3. I fit K-means on the training split only and assigned validation/test samples using fixed train centroids.
4. I evaluated cluster quality with unsupervised metrics.
5. I evaluated pseudo-label learnability with prototype-based few-shot classification.
6. I visually inspected clusters and found that raw clusters were driven by lighting/context.
7. I added balanced ITA pseudo-labels to counter cluster imbalance.
8. I added lighting bias analysis to quantify the difference between raw-image brightness and skin-region brightness.
9. I used the MSKCC Skin Tone Labeling Dataset as external validation to test whether ITA itself is meaningful under controlled measurements.

## Repository Structure

```text
.
├── config.py
├── data/
│   └── mskcc/                         # Small MSKCC CSV files only
├── results/
│   ├── metadata.csv
│   ├── color_features.csv
│   ├── raw_vs_segmented_k_sweep_metrics.csv
│   ├── pseudolabels_k6.csv
│   ├── pseudolabels_k6_raw.csv
│   ├── pseudolabels_k6_balanced_ita_seg.csv
│   ├── pseudolabels_k6_balanced_ita_raw.csv
│   ├── seg_balanced_ita_lighting_bias_summary.csv
│   ├── raw_balanced_ita_lighting_bias_summary.csv
│   ├── mskcc/
│   └── visual_clusters/
└── src/
    ├── build_metadata.py
    ├── color_features.py
    ├── cluster.py
    ├── create_pseudolabels.py
    ├── create_pseudolabels_raw.py
    ├── create_balanced_pseudolabels.py
    ├── fewshot_eval.py
    ├── lighting_bias_analysis.py
    ├── mskcc_validation.py
    └── visualize_clusters.py
```

## Code Files and What They Do

| File | Purpose | Main Output |
|---|---|---|
| `config.py` | Stores paths to the CV segmentation project, raw images, masks, splits, and results folder. | Shared constants |
| `src/build_metadata.py` | Builds a table linking each image ID to image path, mask path, and train/val/test split. | `results/metadata.csv` |
| `src/color_features.py` | Extracts raw and segmented CIELAB/ITA statistics from each image. | `results/color_features.csv` |
| `src/cluster.py` | Runs raw vs segmented K-means sweeps for K = 3, 6, 8, 10 and computes clustering metrics. | `results/raw_vs_segmented_k_sweep_metrics.csv` |
| `src/create_pseudolabels.py` | Creates K=6 pseudo-labels from segmented features. | `results/pseudolabels_k6.csv` |
| `src/create_pseudolabels_raw.py` | Creates K=6 pseudo-labels from raw features. | `results/pseudolabels_k6_raw.csv` |
| `src/create_balanced_pseudolabels.py` | Creates balanced ITA quantile pseudo-labels for both raw and segmented pipelines. | `results/pseudolabels_k6_balanced_ita_*.csv` |
| `src/fewshot_eval.py` | Runs prototype-based 1-shot, 3-shot, and 5-shot evaluation on pseudo-labels. | Printed metrics and confusion matrices |
| `src/visualize_clusters.py` | Creates cluster contact sheets for visual inspection. | `results/visual_clusters/` |
| `src/lighting_bias_analysis.py` | Quantifies how raw-image brightness differs from segmented skin-region brightness. | `*_lighting_bias_summary.csv` |
| `src/mskcc_validation.py` | Uses MSKCC CSVs to validate colorimeter ITA vs expert labels and image ITA vs colorimeter ITA. | `results/mskcc/` |

Link to the data and evaluation results of the visual clusters can be found here: https://drive.google.com/drive/folders/1OMO742u9q0Ec5qUi802TA0iFPPArwYKa?usp=sharing

## Note

The pipeline avoids data leakage by separating deterministic feature extraction from learned model fitting.

Feature extraction is applied to every image independently:

```text
(image, mask) -> CIELAB/ITA features
```

This step learns no parameters. It does not use information from other images or from labels. Therefore, applying the same fixed feature extraction to train, validation, and test is analogous to resizing or color conversion.

The learned steps are handled split-safely:

```text
train features -> fit scaler and K-means -> learned centroids
val/test features -> transform using train scaler -> assign nearest train centroid
```

Validation and test samples never influence the centroids. They are only assigned to the already learned clusters, analogous to prediction in supervised learning.


## Metrics and Why They Matter

| Metric | Where Used | Meaning |
|---|---|---|
| Silhouette score | Clustering | Higher means points are closer to their own cluster than others. |
| Davies-Bouldin index | Clustering | Lower means less overlap between clusters. |
| Calinski-Harabasz score | Clustering | Higher means more compact and separated clusters. |
| Accuracy | Few-shot | Overall fraction of correct pseudo-label predictions. |
| Macro F1 | Few-shot | Treats all clusters equally, useful when pseudo-labels are imbalanced. |
| Balanced accuracy | Few-shot | Average recall across classes, useful for imbalance. |
| Top-2 / Top-3 accuracy | Few-shot soft ranking | Tests whether the correct cluster is near the top, relevant for recommendation-style shade matching. |
| Pearson/Spearman correlation | MSKCC validation | Tests whether ITA aligns with expert/device labels. |
| ARI/NMI | MSKCC clustering agreement | Tests whether unsupervised clusters align with external labels. |

## Main Results

### Dataset Counts After Feature Extraction

| Split | Images |
|---|---:|
| Train | 24,182 |
| Validation | 2,993 |
| Test | 2,824 |
| Total | 29,999 |

One image was skipped because its mask/features were unusable.

### Raw vs Segmented K-Means Sweep

| K | Pipeline | Test Silhouette ↑ | Test DB ↓ | Test CH ↑ |
|---:|---|---:|---:|---:|
| 3 | Segmented | 0.284 | 1.184 | 1080.52 |
| 3 | Raw | 0.263 | 1.284 | 996.91 |
| 6 | Segmented | 0.218 | 1.188 | 874.07 |
| 6 | Raw | 0.222 | 1.261 | 821.61 |
| 8 | Segmented | 0.219 | 1.132 | 807.92 |
| 8 | Raw | 0.202 | 1.227 | 732.63 |
| 10 | Segmented | 0.211 | 1.151 | 746.94 |
| 10 | Raw | 0.204 | 1.230 | 677.92 |

Segmented features generally improved Davies-Bouldin and Calinski-Harabasz, suggesting better compactness/separation. However, visual inspection showed that this did not guarantee perceptual skin-tone correctness.

### Prototype Few-Shot Results on K=6 K-Means Pseudo-Labels

| Shot | Pipeline | Accuracy | Macro F1 | Balanced Acc | Top-2 | Top-3 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | Segmented | 0.399 | 0.442 | 0.419 | 0.687 | 0.827 |
| 1 | Raw | 0.624 | 0.620 | 0.612 | 0.864 | 0.964 |
| 3 | Segmented | 0.669 | 0.685 | 0.676 | 0.850 | 0.978 |
| 3 | Raw | 0.693 | 0.702 | 0.697 | 0.911 | 0.975 |
| 5 | Segmented | 0.654 | 0.685 | 0.667 | 0.853 | 0.972 |
| 5 | Raw | 0.705 | 0.710 | 0.711 | 0.912 | 0.971 |

Both pipelines beat 6-way chance (16.7%), so the pseudo-labels are learnable. Raw performed better because it captured global image properties like brightness and contrast, but visual inspection showed that these were not necessarily true skin-tone cues.

### Balanced ITA Pseudo-Labels

Balanced ITA bins were created to counter K-means imbalance collapse. The training split became nearly perfectly balanced across six bins.

| Pipeline | Train Bin Counts |
|---|---|
| Segmented ITA | 4031, 4030, 4030, 4030, 4030, 4031 |
| Raw ITA | 4031, 4030, 4030, 4030, 4030, 4031 |

This fixed representation coverage, but visual inspection still showed that simple image-derived ITA did not perfectly align with perceived skin tone.

### Lighting Bias Analysis

The strongest quantitative finding came from comparing raw-image brightness and segmented skin brightness.

| Pipeline | Key Pattern | Interpretation |
|---|---|---|
| Segmented balanced ITA | `seg_L_mean` std about 2.8-5.4, while `raw_L_mean` std about 11-12 | Skin-region lightness is consistent inside bins, but whole-image brightness is noisy. |
| Raw balanced ITA | `raw_L_mean` std about 3-6, while `seg_L_mean` std about 9-11 | Raw bins are consistent in global image brightness, but inconsistent in actual skin brightness. |

This shows that raw pseudo-labels are driven by global lighting/exposure, while segmented features better isolate skin-region signal.

### MSKCC External Validation

MSKCC was used as an external diagnostic dataset, not merged into the CelebA pipeline. It tests whether ITA is meaningful under controlled/device measurement.

| Comparison | Pearson r | Spearman r | Interpretation |
|---|---:|---:|---|
| Colorimeter ITA vs FST | -0.790 | -0.803 | Strong relationship. |
| Colorimeter ITA vs MST Rater 1 | -0.912 | -0.898 | Very strong relationship. |
| Colorimeter ITA vs MST Rater 2 | -0.932 | -0.918 | Very strong relationship. |
| Image-extracted ITA vs Colorimeter ITA | -0.015 | -0.064 | Near-zero relationship. |

The negative sign is expected: as FST/MST increase toward darker skin types, ITA decreases.

## Final Conclusion

The pipeline successfully learned structure from image-derived CIELAB/ITA features, and those pseudo-labels were learnable under few-shot constraints. However, the structure learned from unconstrained images was not reliably true skin tone. Raw features were more learnable because they encoded global image brightness and context. Segmented features better isolated skin-region color, but segmentation alone could not solve illumination noise. MSKCC validation showed that colorimeter-derived ITA strongly aligns with expert skin tone labels, while image-extracted ITA does not. Therefore, the limitation is not the colorimetric representation itself, but the reliability of extracting that representation from uncontrolled images.
