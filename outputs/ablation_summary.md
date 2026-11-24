# SPECTRE / SAMv2 Fusion Ablation

Protocol: 5-fold StratifiedKFold x 5 repeats (seeds 42..46, 25 folds total), StandardScaler fit strictly inside each training fold, LogReg (C=1.0, class_weight=balanced, max_iter=5000) and LinearSVM (C=0.01, class_weight=balanced, max_iter=20000) -- identical to evaluate_embeddings.py. PCA (where used) is fit inside each training fold only.

**Skipped:** Anatomical_fusion_raw / multiclass -- the 5-class `multiclass` task combined with `class_weight="balanced"` (CT-4 has only 2 samples, so lbfgs sees an extreme per-sample weight) made LogReg converge pathologically slowly on this config's 8736 raw dimensions; `binary` and `multiclass4` still ran with the exact, unmodified protocol for every configuration.

## Feature configurations

| Config | Dims |
|---|---|
| SPECTRE_global | 1080 |
| SAMv2_only | 1176 |
| Global_fusion | 1464 |
| Anatomical_fusion_raw | 8736 |
| Anatomical_fusion_PCA64 | 2384 |
| Anatomical_fusion_PCA128 | 2512 |

## Full comparison (fold-level mean +/- std over 25 folds)

| Config | Task | Model | Macro-F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|---|
| SPECTRE_global | binary | LogReg | 0.861 +/- 0.022 | 0.946 +/- 0.016 | 0.893 +/- 0.019 |
| SPECTRE_global | binary | LinearSVM | 0.853 +/- 0.024 | 0.949 +/- 0.015 | 0.882 +/- 0.022 |
| SPECTRE_global | multiclass4 | LogReg | 0.518 +/- 0.044 | 0.820 +/- 0.027 | 0.660 +/- 0.033 |
| SPECTRE_global | multiclass4 | LinearSVM | 0.551 +/- 0.051 | 0.839 +/- 0.028 | 0.713 +/- 0.028 |
| SPECTRE_global | multiclass | LogReg | 0.411 +/- 0.032 | 0.802 +/- 0.039 | 0.656 +/- 0.028 |
| SPECTRE_global | multiclass | LinearSVM | 0.435 +/- 0.037 | 0.841 +/- 0.027 | 0.711 +/- 0.026 |
| SAMv2_only | binary | LogReg | 0.593 +/- 0.036 | 0.661 +/- 0.035 | 0.700 +/- 0.029 |
| SAMv2_only | binary | LinearSVM | 0.590 +/- 0.033 | 0.660 +/- 0.035 | 0.681 +/- 0.028 |
| SAMv2_only | multiclass4 | LogReg | 0.320 +/- 0.026 | 0.633 +/- 0.022 | 0.512 +/- 0.027 |
| SAMv2_only | multiclass4 | LinearSVM | 0.316 +/- 0.026 | 0.623 +/- 0.021 | 0.502 +/- 0.026 |
| SAMv2_only | multiclass | LogReg | 0.257 +/- 0.026 | 0.640 +/- 0.036 | 0.512 +/- 0.024 |
| SAMv2_only | multiclass | LinearSVM | 0.254 +/- 0.023 | 0.621 +/- 0.024 | 0.500 +/- 0.028 |
| Global_fusion | binary | LogReg | 0.837 +/- 0.030 | 0.936 +/- 0.018 | 0.879 +/- 0.023 |
| Global_fusion | binary | LinearSVM | 0.860 +/- 0.026 | 0.947 +/- 0.017 | 0.890 +/- 0.022 |
| Global_fusion | multiclass4 | LogReg | 0.490 +/- 0.034 | 0.802 +/- 0.030 | 0.654 +/- 0.033 |
| Global_fusion | multiclass4 | LinearSVM | 0.533 +/- 0.037 | 0.813 +/- 0.033 | 0.694 +/- 0.029 |
| Global_fusion | multiclass | LogReg | 0.395 +/- 0.038 | 0.786 +/- 0.038 | 0.656 +/- 0.035 |
| Global_fusion | multiclass | LinearSVM | 0.417 +/- 0.038 | 0.814 +/- 0.031 | 0.688 +/- 0.030 |
| Anatomical_fusion_raw | binary | LogReg | 0.853 +/- 0.028 | 0.941 +/- 0.016 | 0.892 +/- 0.023 |
| Anatomical_fusion_raw | binary | LinearSVM | 0.807 +/- 0.025 | 0.920 +/- 0.020 | 0.846 +/- 0.022 |
| Anatomical_fusion_raw | multiclass4 | LogReg | 0.533 +/- 0.038 | 0.817 +/- 0.029 | 0.709 +/- 0.024 |
| Anatomical_fusion_raw | multiclass4 | LinearSVM | 0.494 +/- 0.035 | 0.778 +/- 0.029 | 0.661 +/- 0.024 |
| Anatomical_fusion_PCA64 | binary | LogReg | 0.839 +/- 0.026 | 0.936 +/- 0.015 | 0.883 +/- 0.021 |
| Anatomical_fusion_PCA64 | binary | LinearSVM | 0.813 +/- 0.033 | 0.918 +/- 0.021 | 0.857 +/- 0.027 |
| Anatomical_fusion_PCA64 | multiclass4 | LogReg | 0.537 +/- 0.038 | 0.814 +/- 0.029 | 0.694 +/- 0.021 |
| Anatomical_fusion_PCA64 | multiclass4 | LinearSVM | 0.499 +/- 0.038 | 0.778 +/- 0.028 | 0.663 +/- 0.021 |
| Anatomical_fusion_PCA64 | multiclass | LogReg | 0.410 +/- 0.036 | 0.778 +/- 0.053 | 0.688 +/- 0.020 |
| Anatomical_fusion_PCA64 | multiclass | LinearSVM | 0.389 +/- 0.032 | 0.769 +/- 0.028 | 0.662 +/- 0.025 |
| Anatomical_fusion_PCA128 | binary | LogReg | 0.842 +/- 0.024 | 0.935 +/- 0.015 | 0.885 +/- 0.019 |
| Anatomical_fusion_PCA128 | binary | LinearSVM | 0.815 +/- 0.031 | 0.918 +/- 0.019 | 0.857 +/- 0.026 |
| Anatomical_fusion_PCA128 | multiclass4 | LogReg | 0.530 +/- 0.044 | 0.814 +/- 0.027 | 0.689 +/- 0.025 |
| Anatomical_fusion_PCA128 | multiclass4 | LinearSVM | 0.489 +/- 0.040 | 0.768 +/- 0.029 | 0.660 +/- 0.025 |
| Anatomical_fusion_PCA128 | multiclass | LogReg | 0.411 +/- 0.038 | 0.772 +/- 0.058 | 0.686 +/- 0.024 |
| Anatomical_fusion_PCA128 | multiclass | LinearSVM | 0.378 +/- 0.025 | 0.772 +/- 0.025 | 0.655 +/- 0.025 |

## Significance vs SPECTRE_global (paired t-test / Wilcoxon on the matched 25 folds)

### Macro-F1, task=multiclass4 (highlighted: this is the main severity-ordinal task)

| Config | Model | Mean diff | t_p | wilcoxon_p | n folds |
|---|---|---|---|---|---|
| SAMv2_only | LogReg | -0.1979 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | LinearSVM | -0.2353 | 0.0000*** | 0.0000*** | 25 |
| Global_fusion | LogReg | -0.0273 | 0.0031** | 0.0009*** | 25 |
| Global_fusion | LinearSVM | -0.0181 | 0.0172* | 0.0173* | 25 |
| Anatomical_fusion_raw | LogReg | +0.0153 | 0.0960 | 0.0667 | 25 |
| Anatomical_fusion_raw | LinearSVM | -0.0571 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | LogReg | +0.0192 | 0.0184* | 0.0342* | 25 |
| Anatomical_fusion_PCA64 | LinearSVM | -0.0517 | 0.0001*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA128 | LogReg | +0.0126 | 0.0983 | 0.1014 | 25 |
| Anatomical_fusion_PCA128 | LinearSVM | -0.0617 | 0.0000*** | 0.0000*** | 25 |

### ROC-AUC, task=binary

| Config | Model | Mean diff | t_p | wilcoxon_p | n folds |
|---|---|---|---|---|---|
| SAMv2_only | LogReg | -0.2845 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | LinearSVM | -0.2890 | 0.0000*** | 0.0000*** | 25 |
| Global_fusion | LogReg | -0.0095 | 0.0000*** | 0.0001*** | 25 |
| Global_fusion | LinearSVM | -0.0025 | 0.0626 | 0.0447* | 25 |
| Anatomical_fusion_raw | LogReg | -0.0052 | 0.0176* | 0.0275* | 25 |
| Anatomical_fusion_raw | LinearSVM | -0.0290 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | LogReg | -0.0102 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | LinearSVM | -0.0319 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA128 | LogReg | -0.0109 | 0.0000*** | 0.0001*** | 25 |
| Anatomical_fusion_PCA128 | LinearSVM | -0.0316 | 0.0000*** | 0.0000*** | 25 |

### All tasks, Macro-F1 (full significance table)

| Config | Task | Model | Mean diff | t_p | wilcoxon_p | n folds |
|---|---|---|---|---|---|---|
| SAMv2_only | binary | LogReg | -0.2687 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | binary | LinearSVM | -0.2631 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | multiclass4 | LogReg | -0.1979 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | multiclass4 | LinearSVM | -0.2353 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | multiclass | LogReg | -0.1540 | 0.0000*** | 0.0000*** | 25 |
| SAMv2_only | multiclass | LinearSVM | -0.1812 | 0.0000*** | 0.0000*** | 25 |
| Global_fusion | binary | LogReg | -0.0246 | 0.0001*** | 0.0002*** | 25 |
| Global_fusion | binary | LinearSVM | +0.0067 | 0.0218* | 0.0370* | 25 |
| Global_fusion | multiclass4 | LogReg | -0.0273 | 0.0031** | 0.0009*** | 25 |
| Global_fusion | multiclass4 | LinearSVM | -0.0181 | 0.0172* | 0.0173* | 25 |
| Global_fusion | multiclass | LogReg | -0.0155 | 0.0809 | 0.0710 | 25 |
| Global_fusion | multiclass | LinearSVM | -0.0177 | 0.0127* | 0.0219* | 25 |
| Anatomical_fusion_raw | binary | LogReg | -0.0083 | 0.0699 | 0.0667 | 25 |
| Anatomical_fusion_raw | binary | LinearSVM | -0.0455 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_raw | multiclass4 | LogReg | +0.0153 | 0.0960 | 0.0667 | 25 |
| Anatomical_fusion_raw | multiclass4 | LinearSVM | -0.0571 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | binary | LogReg | -0.0221 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | binary | LinearSVM | -0.0399 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | multiclass4 | LogReg | +0.0192 | 0.0184* | 0.0342* | 25 |
| Anatomical_fusion_PCA64 | multiclass4 | LinearSVM | -0.0517 | 0.0001*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA64 | multiclass | LogReg | -0.0001 | 0.9876 | 0.9368 | 25 |
| Anatomical_fusion_PCA64 | multiclass | LinearSVM | -0.0462 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA128 | binary | LogReg | -0.0192 | 0.0006*** | 0.0009*** | 25 |
| Anatomical_fusion_PCA128 | binary | LinearSVM | -0.0381 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA128 | multiclass4 | LogReg | +0.0126 | 0.0983 | 0.1014 | 25 |
| Anatomical_fusion_PCA128 | multiclass4 | LinearSVM | -0.0617 | 0.0000*** | 0.0000*** | 25 |
| Anatomical_fusion_PCA128 | multiclass | LogReg | +0.0004 | 0.9530 | 0.7712 | 25 |
| Anatomical_fusion_PCA128 | multiclass | LinearSVM | -0.0575 | 0.0000*** | 0.0000*** | 25 |

p < 0.05 `*`, p < 0.01 `**`, p < 0.001 `***`. Mean diff = config - SPECTRE_global (positive = fusion variant better).
