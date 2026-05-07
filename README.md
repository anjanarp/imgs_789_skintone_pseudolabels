# Segmentation-Aware Colorimetry for Skin Appearance Pseudo-Labeling Under Data Scarcity

This repository implements a segmentation-aware colorimetry pipeline for studying whether image-derived skin appearance pseudo-labels can be learned by a CNN under progressively reduced training data availability. The project began as a downstream extension of my computer vision segmentation project and evolved into an investigation of pseudo-label stability, data scarcity, lighting bias, and external colorimetry validation.

The core idea is to generate pseudo-labels from image-derived CIELAB/ITA skin appearance features, then quantify how CNN classification performance changes as increasing amounts of training data are removed from every pseudo-label cluster.

---

# Core Research Question

Can raw or segmentation-aware image-derived color features produce meaningful skin appearance pseudo-labels, and how robust are those pseudo-labels under controlled training data scarcity?

More specifically:

- Does segmentation improve pseudo-label quality compared to raw-image features?
- Does balancing ITA distributions improve learnability?
- How stable are pseudo-labels as training data is progressively reduced?
- Are image-derived ITA measurements aligned with externally validated skin tone measurements?

---

# High-Level Project Story

1. I started from my CV project, which produced conservative engineered skin masks for CelebA faces.
2. I extracted CIELAB/ITA features from both raw images and segmented skin-only regions.
3. I fit K-means on the training split only and assigned validation/test samples using fixed train centroids.
4. I created balanced ITA pseudo-labels to counter cluster imbalance collapse.
5. I trained CNN classifiers to predict pseudo-labels from face images.
6. I simulated data scarcity by randomly removing 5% to 50% of samples from every pseudo-label cluster.
7. I evaluated CNN performance on a fixed held-out test set using accuracy, balanced accuracy, and macro F1.
8. I visually inspected clusters and analyzed lighting bias between raw-image brightness and segmented skin-region brightness.
9. I used the MSKCC Skin Tone Labeling Dataset as an external validation benchmark to test whether ITA itself is meaningful under controlled measurements.

---

# Repository Structure

```text
.
├── config.py
├── data/
│   └── mskcc/
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
│   ├── data_scarcity_cnn_*/
│   ├── mskcc/
│   └── visual_clusters/
└── src/
    ├── build_metadata.py
    ├── color_features.py
    ├── cluster.py
    ├── create_pseudolabels.py
    ├── create_pseudolabels_raw.py
    ├── create_balanced_pseudolabels.py
    ├── data_scarcity_cnn.py
    ├── lighting_bias_analysis.py
    ├── mskcc_validation.py
    └── visualize_clusters.py
```

---

# Code Files and What They Do

| File | Purpose | Main Output |
|---|---|---|
| `config.py` | Stores paths to raw images, masks, splits, and results folders. | Shared constants |
| `src/build_metadata.py` | Builds image/mask/split metadata table. | `results/metadata.csv` |
| `src/color_features.py` | Extracts raw and segmented CIELAB/ITA statistics. | `results/color_features.csv` |
| `src/cluster.py` | Runs raw vs segmented K-means sweeps and computes clustering metrics. | `results/raw_vs_segmented_k_sweep_metrics.csv` |
| `src/create_pseudolabels.py` | Creates segmented K-means pseudo-labels. | `results/pseudolabels_k6.csv` |
| `src/create_pseudolabels_raw.py` | Creates raw-image K-means pseudo-labels. | `results/pseudolabels_k6_raw.csv` |
| `src/create_balanced_pseudolabels.py` | Creates balanced ITA quantile pseudo-labels. | `results/pseudolabels_k6_balanced_ita_*.csv` |
| `src/data_scarcity_cnn.py` | Trains CNNs under randomized per-cluster data scarcity conditions. | CNN metrics + scarcity curves |
| `src/visualize_clusters.py` | Creates cluster contact sheets for qualitative inspection. | `results/visual_clusters/` |
| `src/lighting_bias_analysis.py` | Quantifies raw-image brightness vs segmented-skin brightness. | `*_lighting_bias_summary.csv` |
| `src/mskcc_validation.py` | Validates ITA using MSKCC expert/colorimeter labels. | `results/mskcc/` |

---

# Data Leakage Defense

The pipeline avoids data leakage by separating deterministic feature extraction from learned model fitting.

Feature extraction is applied independently to every image:

```text
(image, mask) -> CIELAB/ITA features
```

This stage learns no parameters.

Learned stages are split-safe:

```text
train features -> fit scaler/K-means -> learned centroids
val/test features -> transform using train scaler -> assign nearest train centroid
```

Validation and test samples never influence learned centroids or balancing thresholds.

---

# Metrics and Why They Matter

| Metric | Meaning |
|---|---|
| Accuracy | Overall pseudo-label classification performance |
| Balanced Accuracy | Average recall across classes, important under imbalance |
| Macro F1 | Treats all pseudo-label classes equally |
| Silhouette Score | Cluster compactness and separation |
| Davies-Bouldin Index | Cluster overlap measure (lower is better) |
| Calinski-Harabasz Score | Cluster compactness/separation |
| Pearson/Spearman Correlation | Agreement with external MSKCC labels |

---

# Dataset Counts

| Split | Images |
|---|---:|
| Train | 24,182 |
| Validation | 2,993 |
| Test | 2,824 |
| Total | 29,999 |

One image was discarded due to unusable features.

---

# Raw vs Segmented K-Means Sweep

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

Segmented features generally improved compactness and separation metrics, suggesting that segmentation isolates more coherent color information than raw-image statistics.

---

# CNN Data Scarcity Experiments

The central experiment trains CNN classifiers on pseudo-label targets while progressively reducing the amount of training data retained from every cluster.

Each experiment:

1. Keeps the validation/test split fixed.
2. Randomly subsamples every pseudo-label cluster independently.
3. Trains the same CNN architecture under each reduced-data condition.
4. Evaluates performance degradation.

---

# CNN Results — Raw Balanced ITA Pseudo-Labels

| Training Data Kept | Accuracy | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 100% | 0.813 | 0.803 | 0.807 |
| 95% | 0.886 | 0.886 | 0.885 |
| 90% | 0.771 | 0.778 | 0.778 |
| 80% | 0.911 | 0.911 | 0.910 |
| 70% | 0.868 | 0.869 | 0.870 |
| 60% | 0.837 | 0.834 | 0.828 |
| 50% | 0.852 | 0.853 | 0.853 |

Raw balanced ITA pseudo-labels were highly learnable even under moderate data reduction, suggesting that the CNN captured strong global appearance signals.

---

# CNN Results — Raw K-Means Pseudo-Labels

| Training Data Kept | Accuracy | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 100% | 0.851 | 0.864 | 0.847 |
| 95% | 0.864 | 0.862 | 0.858 |
| 90% | 0.774 | 0.802 | 0.773 |
| 80% | 0.745 | 0.756 | 0.738 |
| 70% | 0.872 | 0.887 | 0.872 |
| 60% | 0.807 | 0.777 | 0.803 |
| 50% | 0.825 | 0.822 | 0.822 |

Raw K-means pseudo-labels also remained learnable, but variability across subsampling levels suggests stronger sensitivity to random cluster composition.

---

# CNN Results — Segmented Balanced ITA Pseudo-Labels

| Training Data Kept | Accuracy | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 100% | 0.623 | 0.623 | 0.619 |
| 95% | 0.477 | 0.473 | 0.465 |
| 90% | 0.723 | 0.723 | 0.718 |
| 80% | 0.685 | 0.686 | 0.678 |
| 70% | 0.697 | 0.696 | 0.693 |
| 60% | 0.591 | 0.590 | 0.589 |
| 50% | 0.688 | 0.686 | 0.686 |

Segmented ITA pseudo-labels were more difficult for the CNN to learn consistently, suggesting that removing contextual brightness cues forces the model to rely more directly on localized skin appearance.

---

# CNN Results — Segmented K-Means Pseudo-Labels

| Training Data Kept | Accuracy | Balanced Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 100% | 0.623 | 0.623 | 0.619 |
| 95% | 0.477 | 0.473 | 0.465 |
| 90% | 0.723 | 0.723 | 0.718 |
| 80% | 0.685 | 0.686 | 0.678 |
| 70% | 0.697 | 0.696 | 0.693 |
| 60% | 0.591 | 0.590 | 0.589 |
| 50% | 0.688 | 0.686 | 0.686 |

The segmented pipelines produced lower but potentially more semantically meaningful performance because segmentation suppresses global image illumination/context artifacts.

---

# Lighting Bias Analysis

The strongest diagnostic finding came from comparing raw-image brightness and segmented-skin brightness.

| Pipeline | Key Pattern | Interpretation |
|---|---|---|
| Segmented balanced ITA | Skin-region brightness stable, raw-image brightness noisy | Segmentation isolates skin-region signal |
| Raw balanced ITA | Raw-image brightness stable, skin brightness noisy | Raw labels are strongly influenced by global illumination/context |

This suggests that raw pseudo-labels are easier for CNNs to learn partly because they encode global image properties unrelated to true skin appearance.

---

# MSKCC External Validation

MSKCC was used as an external diagnostic benchmark.

| Comparison | Pearson r | Spearman r |
|---|---:|---:|
| Colorimeter ITA vs FST | -0.790 | -0.803 |
| Colorimeter ITA vs MST Rater 1 | -0.912 | -0.898 |
| Colorimeter ITA vs MST Rater 2 | -0.932 | -0.918 |
| Image-derived ITA vs Colorimeter ITA | -0.015 | -0.064 |

These results show that ITA itself is meaningful under controlled measurement conditions, while extracting ITA reliably from unconstrained consumer-style images remains difficult.

---

# Final Conclusion

This project demonstrates that image-derived skin appearance pseudo-labels are learnable by CNN classifiers even under moderate training data scarcity. However, the most learnable pseudo-labels were often driven by global image brightness and contextual illumination rather than true localized skin appearance.

Segmentation-aware pipelines reduced these shortcuts by isolating skin-region information, but this also made the classification problem harder and more sensitive to noise introduced by uncontrolled imaging conditions.

The external MSKCC validation further showed that ITA is meaningful when measured under controlled conditions using calibrated instrumentation, but extracting reliable skin appearance measurements from unconstrained images remains a fundamentally difficult problem.
