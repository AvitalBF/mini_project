# Stage 3 Follow-up: Clean Anatomical Routing (SPECTRE-only regional fusion)

## Protocol

- Source: `outputs/spectre_samv2_fusion.npz` (1110 scans; no re-extraction). **No SAMv2 feature vectors are used.** SAMv2 served only as the anatomical router that produced the six regional SPECTRE embeddings (`spectre_regional`, 6 x 1080).
- Outer CV: 5-fold StratifiedKFold x 5 repeats (seeds 42..46, 25 folds), identical splits to `evaluate_embeddings.py` / `ablation_fusion.py`.
- Tasks: `binary` (CT-0 vs CT-1..4) and `multiclass4` (CT-0 / CT-1 / CT-2 / CT-3+4). The 5-class `multiclass` task was intentionally skipped.
- Classifiers: LogReg (class_weight=balanced, max_iter=5000), LinearSVM (class_weight=balanced, max_iter=20000).
- **Regime A (fixed):** LogReg C=1.0, LinearSVM C=0.01 (original baseline protocol).
- **Regime B (tuned C):** C in [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0] chosen by inner 3-fold StratifiedKFold (macro-F1) inside every outer training fold, for every configuration including the baseline.
- Late fusion: alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0] chosen by inner 3-fold StratifiedKFold (macro-F1) inside every outer training fold, in both regimes; ties go to the larger alpha.
- Leakage control: StandardScaler, PCA, C and alpha are all fitted on the outer training fold only (and refit on every inner split). The only pre-CV transform, per-scan mean-centering of the six zone embeddings, uses no information from other scans.
- Wall-clock runtime: 1196 s.

## Configurations

| Config | Dims | Description |
|---|---|---|
| SPECTRE_global | 1080 | Whole-scan SPECTRE embedding (baseline) |
| SPECTRE_regional_PCA64 | 64 | Regional SPECTRE (6x1080) -> in-fold PCA(64); standalone regional signal |
| Clean_Anat_PCA32_L2 | 1080 + 32 | global + PCA32(regional); per-block StandardScaler + L2 |
| Clean_Anat_PCA64_L2 | 1080 + 64 | global + PCA64(regional); per-block StandardScaler + L2 |
| Clean_Residual_PCA32_L2 | 1080 + 32 | global + PCA32(zone residuals, mean-centered per scan); per-block Scaler + L2 |
| Clean_Residual_PCA64_L2 | 1080 + 64 | global + PCA64(zone residuals, mean-centered per scan); per-block Scaler + L2 |
| Late_Fusion_Prob | 1080 / 64 (two models) | alpha * s(global model) + (1 - alpha) * s(regional PCA64 model); alpha via inner CV |

## Regime A: fixed C (LogReg C=1.0, LinearSVM C=0.01)

Fold-level mean +/- std over 25 folds. Delta = paired mean difference in macro-F1 vs `SPECTRE_global` (same regime, task, model); p-values: paired t-test / Wilcoxon.

### Task: `binary`

| Config | Model | Macro-F1 | ROC-AUC | Accuracy | dMacro-F1 | t p | Wilcoxon p | Selected hparams |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_global | LogReg | 0.861 +/- 0.023 | 0.946 +/- 0.016 | 0.893 +/- 0.020 | (ref) |  |  | fixed |
| SPECTRE_regional_PCA64 | LogReg | 0.828 +/- 0.023 | 0.937 +/- 0.017 | 0.864 +/- 0.021 | -0.0332 | 0.0000*** | 0.0000*** | fixed |
| Clean_Anat_PCA32_L2 | LogReg | 0.850 +/- 0.028 | 0.947 +/- 0.016 | 0.879 +/- 0.025 | -0.0113 | 0.0048** | 0.0051** | fixed |
| Clean_Anat_PCA64_L2 | LogReg | 0.846 +/- 0.026 | 0.946 +/- 0.016 | 0.877 +/- 0.023 | -0.0150 | 0.0001*** | 0.0001*** | fixed |
| Clean_Residual_PCA32_L2 | LogReg | 0.845 +/- 0.026 | 0.945 +/- 0.016 | 0.875 +/- 0.023 | -0.0163 | 0.0000*** | 0.0001*** | fixed |
| Clean_Residual_PCA64_L2 | LogReg | 0.844 +/- 0.026 | 0.943 +/- 0.017 | 0.875 +/- 0.023 | -0.0170 | 0.0001*** | 0.0001*** | fixed |
| Late_Fusion_Prob | LogReg | 0.862 +/- 0.022 | 0.953 +/- 0.015 | 0.893 +/- 0.020 | +0.0009 | 0.5857 | 0.5677 | C_g: mode 1 (25/25); C_r: mode 1 (25/25); alpha: mode 0.5 (11/25) |
| SPECTRE_global | LinearSVM | 0.853 +/- 0.024 | 0.949 +/- 0.015 | 0.882 +/- 0.022 | (ref) |  |  | fixed |
| SPECTRE_regional_PCA64 | LinearSVM | 0.796 +/- 0.026 | 0.931 +/- 0.017 | 0.829 +/- 0.026 | -0.0568 | 0.0000*** | 0.0000*** | fixed |
| Clean_Anat_PCA32_L2 | LinearSVM | 0.828 +/- 0.026 | 0.944 +/- 0.017 | 0.857 +/- 0.024 | -0.0253 | 0.0000*** | 0.0000*** | fixed |
| Clean_Anat_PCA64_L2 | LinearSVM | 0.826 +/- 0.027 | 0.943 +/- 0.017 | 0.855 +/- 0.025 | -0.0270 | 0.0000*** | 0.0000*** | fixed |
| Clean_Residual_PCA32_L2 | LinearSVM | 0.828 +/- 0.025 | 0.943 +/- 0.016 | 0.857 +/- 0.024 | -0.0252 | 0.0000*** | 0.0000*** | fixed |
| Clean_Residual_PCA64_L2 | LinearSVM | 0.825 +/- 0.025 | 0.942 +/- 0.017 | 0.854 +/- 0.023 | -0.0279 | 0.0000*** | 0.0000*** | fixed |
| Late_Fusion_Prob | LinearSVM | 0.852 +/- 0.026 | 0.952 +/- 0.015 | 0.880 +/- 0.023 | -0.0013 | 0.3476 | 0.3720 | C_g: mode 0.01 (25/25); C_r: mode 0.01 (25/25); alpha: mode 0.8 (7/25) |

### Task: `multiclass4`

| Config | Model | Macro-F1 | ROC-AUC | Accuracy | dMacro-F1 | t p | Wilcoxon p | Selected hparams |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_global | LogReg | 0.518 +/- 0.045 | 0.820 +/- 0.027 | 0.660 +/- 0.033 | (ref) |  |  | fixed |
| SPECTRE_regional_PCA64 | LogReg | 0.479 +/- 0.036 | 0.789 +/- 0.029 | 0.595 +/- 0.031 | -0.0385 | 0.0005*** | 0.0001*** | fixed |
| Clean_Anat_PCA32_L2 | LogReg | 0.472 +/- 0.029 | 0.814 +/- 0.025 | 0.574 +/- 0.024 | -0.0457 | 0.0000*** | 0.0000*** | fixed |
| Clean_Anat_PCA64_L2 | LogReg | 0.488 +/- 0.030 | 0.821 +/- 0.026 | 0.597 +/- 0.024 | -0.0299 | 0.0001*** | 0.0001*** | fixed |
| Clean_Residual_PCA32_L2 | LogReg | 0.468 +/- 0.029 | 0.811 +/- 0.027 | 0.568 +/- 0.025 | -0.0499 | 0.0000*** | 0.0000*** | fixed |
| Clean_Residual_PCA64_L2 | LogReg | 0.466 +/- 0.027 | 0.802 +/- 0.027 | 0.575 +/- 0.025 | -0.0524 | 0.0000*** | 0.0000*** | fixed |
| Late_Fusion_Prob | LogReg | 0.523 +/- 0.053 | 0.824 +/- 0.023 | 0.671 +/- 0.035 | +0.0045 | 0.4151 | 0.3173 | C_g: mode 1 (25/25); C_r: mode 1 (25/25); alpha: mode 0.6 (11/25) |
| SPECTRE_global | LinearSVM | 0.551 +/- 0.051 | 0.839 +/- 0.028 | 0.713 +/- 0.028 | (ref) |  |  | fixed |
| SPECTRE_regional_PCA64 | LinearSVM | 0.491 +/- 0.037 | 0.800 +/- 0.025 | 0.655 +/- 0.027 | -0.0599 | 0.0000*** | 0.0000*** | fixed |
| Clean_Anat_PCA32_L2 | LinearSVM | 0.539 +/- 0.053 | 0.840 +/- 0.027 | 0.719 +/- 0.028 | -0.0120 | 0.1475 | 0.2099 | fixed |
| Clean_Anat_PCA64_L2 | LinearSVM | 0.530 +/- 0.052 | 0.845 +/- 0.026 | 0.717 +/- 0.029 | -0.0205 | 0.0104* | 0.0147* | fixed |
| Clean_Residual_PCA32_L2 | LinearSVM | 0.520 +/- 0.050 | 0.838 +/- 0.025 | 0.709 +/- 0.032 | -0.0312 | 0.0021** | 0.0038** | fixed |
| Clean_Residual_PCA64_L2 | LinearSVM | 0.516 +/- 0.049 | 0.839 +/- 0.026 | 0.709 +/- 0.029 | -0.0347 | 0.0005*** | 0.0002*** | fixed |
| Late_Fusion_Prob | LinearSVM | 0.550 +/- 0.056 | 0.843 +/- 0.026 | 0.717 +/- 0.031 | -0.0012 | 0.7935 | 1.0000 | C_g: mode 0.01 (25/25); C_r: mode 0.01 (25/25); alpha: mode 0.5 (5/25) |

### ROC-AUC and accuracy significance vs `SPECTRE_global` (A_fixed)

| Config | Task | Model | dROC-AUC | t p | Wilcoxon p | dAccuracy | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_regional_PCA64 | binary | LogReg | -0.0090 | 0.0009*** | 0.0010** | -0.0286 | 0.0000*** | 0.0000*** |
| SPECTRE_regional_PCA64 | binary | LinearSVM | -0.0183 | 0.0000*** | 0.0001*** | -0.0532 | 0.0000*** | 0.0000*** |
| SPECTRE_regional_PCA64 | multiclass4 | LogReg | -0.0310 | 0.0000*** | 0.0001*** | -0.0650 | 0.0000*** | 0.0000*** |
| SPECTRE_regional_PCA64 | multiclass4 | LinearSVM | -0.0388 | 0.0000*** | 0.0000*** | -0.0580 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA32_L2 | binary | LogReg | +0.0012 | 0.4322 | 0.5965 | -0.0135 | 0.0001*** | 0.0003*** |
| Clean_Anat_PCA32_L2 | binary | LinearSVM | -0.0059 | 0.0003*** | 0.0005*** | -0.0256 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA32_L2 | multiclass4 | LogReg | -0.0059 | 0.1387 | 0.1563 | -0.0859 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA32_L2 | multiclass4 | LinearSVM | +0.0015 | 0.6428 | 0.5965 | +0.0059 | 0.2442 | 0.2352 |
| Clean_Anat_PCA64_L2 | binary | LogReg | -0.0001 | 0.9351 | 0.6272 | -0.0162 | 0.0000*** | 0.0001*** |
| Clean_Anat_PCA64_L2 | binary | LinearSVM | -0.0063 | 0.0001*** | 0.0001*** | -0.0268 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA64_L2 | multiclass4 | LogReg | +0.0014 | 0.7359 | 0.9158 | -0.0636 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA64_L2 | multiclass4 | LinearSVM | +0.0060 | 0.0597 | 0.0516 | +0.0043 | 0.4188 | 0.4231 |
| Clean_Residual_PCA32_L2 | binary | LogReg | -0.0007 | 0.5808 | 0.4273 | -0.0173 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA32_L2 | binary | LinearSVM | -0.0067 | 0.0000*** | 0.0002*** | -0.0250 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA32_L2 | multiclass4 | LogReg | -0.0085 | 0.0320* | 0.0160* | -0.0926 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA32_L2 | multiclass4 | LinearSVM | -0.0007 | 0.8392 | 0.8532 | -0.0038 | 0.5422 | 0.5445 |
| Clean_Residual_PCA64_L2 | binary | LogReg | -0.0024 | 0.0766 | 0.0918 | -0.0180 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA64_L2 | binary | LinearSVM | -0.0075 | 0.0000*** | 0.0000*** | -0.0277 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA64_L2 | multiclass4 | LogReg | -0.0171 | 0.0003*** | 0.0005*** | -0.0852 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA64_L2 | multiclass4 | LinearSVM | +0.0002 | 0.9489 | 0.8740 | -0.0034 | 0.5336 | 0.5105 |
| Late_Fusion_Prob | binary | LogReg | +0.0073 | 0.0000*** | 0.0001*** | +0.0002 | 0.8995 | 0.8695 |
| Late_Fusion_Prob | binary | LinearSVM | +0.0021 | 0.0001*** | 0.0003*** | -0.0018 | 0.1344 | 0.0866 |
| Late_Fusion_Prob | multiclass4 | LogReg | +0.0048 | 0.0668 | 0.1300 | +0.0106 | 0.0087** | 0.0106* |
| Late_Fusion_Prob | multiclass4 | LinearSVM | +0.0043 | 0.0095** | 0.0152* | +0.0043 | 0.1349 | 0.1306 |

## Regime B: inner-CV tuned C

Fold-level mean +/- std over 25 folds. Delta = paired mean difference in macro-F1 vs `SPECTRE_global` (same regime, task, model); p-values: paired t-test / Wilcoxon.

### Task: `binary`

| Config | Model | Macro-F1 | ROC-AUC | Accuracy | dMacro-F1 | t p | Wilcoxon p | Selected hparams |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_global | LogReg | 0.858 +/- 0.022 | 0.947 +/- 0.016 | 0.889 +/- 0.019 | (ref) |  |  | C: mode 1 (11/25) |
| SPECTRE_regional_PCA64 | LogReg | 0.833 +/- 0.025 | 0.937 +/- 0.017 | 0.869 +/- 0.023 | -0.0252 | 0.0000*** | 0.0000*** | C: mode 10 (21/25) |
| Clean_Anat_PCA32_L2 | LogReg | 0.857 +/- 0.025 | 0.947 +/- 0.016 | 0.887 +/- 0.022 | -0.0003 | 0.9191 | 0.9515 | C: mode 10 (19/25) |
| Clean_Anat_PCA64_L2 | LogReg | 0.851 +/- 0.026 | 0.945 +/- 0.017 | 0.881 +/- 0.023 | -0.0072 | 0.0332* | 0.0397* | C: mode 1 (15/25) |
| Clean_Residual_PCA32_L2 | LogReg | 0.847 +/- 0.022 | 0.944 +/- 0.016 | 0.878 +/- 0.019 | -0.0106 | 0.0005*** | 0.0006*** | C: mode 1 (15/25) |
| Clean_Residual_PCA64_L2 | LogReg | 0.844 +/- 0.026 | 0.943 +/- 0.017 | 0.875 +/- 0.023 | -0.0133 | 0.0004*** | 0.0003*** | C: mode 1 (25/25) |
| Late_Fusion_Prob | LogReg | 0.862 +/- 0.023 | 0.952 +/- 0.016 | 0.891 +/- 0.020 | +0.0038 | 0.0764 | 0.0944 | C_g: mode 1 (11/25); C_r: mode 10 (21/25); alpha: mode 0.5 (7/25) |
| SPECTRE_global | LinearSVM | 0.858 +/- 0.021 | 0.946 +/- 0.015 | 0.889 +/- 0.018 | (ref) |  |  | C: mode 0.1 (24/25) |
| SPECTRE_regional_PCA64 | LinearSVM | 0.834 +/- 0.023 | 0.939 +/- 0.017 | 0.869 +/- 0.021 | -0.0237 | 0.0000*** | 0.0000*** | C: mode 10 (14/25) |
| Clean_Anat_PCA32_L2 | LinearSVM | 0.859 +/- 0.024 | 0.947 +/- 0.015 | 0.889 +/- 0.021 | +0.0017 | 0.5187 | 0.3313 | C: mode 10 (18/25) |
| Clean_Anat_PCA64_L2 | LinearSVM | 0.850 +/- 0.024 | 0.945 +/- 0.016 | 0.880 +/- 0.022 | -0.0082 | 0.0225* | 0.0110* | C: mode 1 (14/25) |
| Clean_Residual_PCA32_L2 | LinearSVM | 0.848 +/- 0.026 | 0.944 +/- 0.017 | 0.879 +/- 0.022 | -0.0096 | 0.0079** | 0.0110* | C: mode 0.1 (10/25) |
| Clean_Residual_PCA64_L2 | LinearSVM | 0.843 +/- 0.027 | 0.943 +/- 0.017 | 0.873 +/- 0.024 | -0.0147 | 0.0004*** | 0.0007*** | C: mode 0.1 (23/25) |
| Late_Fusion_Prob | LinearSVM | 0.864 +/- 0.024 | 0.954 +/- 0.015 | 0.894 +/- 0.021 | +0.0068 | 0.0020** | 0.0030** | C_g: mode 0.1 (24/25); C_r: mode 10 (14/25); alpha: mode 0.5 (16/25) |

### Task: `multiclass4`

| Config | Model | Macro-F1 | ROC-AUC | Accuracy | dMacro-F1 | t p | Wilcoxon p | Selected hparams |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_global | LogReg | 0.516 +/- 0.047 | 0.819 +/- 0.025 | 0.652 +/- 0.045 | (ref) |  |  | C: mode 1 (14/25) |
| SPECTRE_regional_PCA64 | LogReg | 0.482 +/- 0.033 | 0.789 +/- 0.029 | 0.600 +/- 0.030 | -0.0345 | 0.0009*** | 0.0008*** | C: mode 10 (17/25) |
| Clean_Anat_PCA32_L2 | LogReg | 0.478 +/- 0.027 | 0.809 +/- 0.026 | 0.583 +/- 0.025 | -0.0384 | 0.0000*** | 0.0000*** | C: mode 10 (16/25) |
| Clean_Anat_PCA64_L2 | LogReg | 0.491 +/- 0.034 | 0.815 +/- 0.026 | 0.608 +/- 0.029 | -0.0248 | 0.0010** | 0.0007*** | C: mode 1 (14/25) |
| Clean_Residual_PCA32_L2 | LogReg | 0.470 +/- 0.033 | 0.810 +/- 0.028 | 0.578 +/- 0.024 | -0.0457 | 0.0000*** | 0.0000*** | C: mode 10 (15/25) |
| Clean_Residual_PCA64_L2 | LogReg | 0.463 +/- 0.029 | 0.793 +/- 0.030 | 0.580 +/- 0.029 | -0.0533 | 0.0000*** | 0.0000*** | C: mode 10 (13/25) |
| Late_Fusion_Prob | LogReg | 0.523 +/- 0.050 | 0.825 +/- 0.024 | 0.665 +/- 0.038 | +0.0069 | 0.1734 | 0.0765 | C_g: mode 1 (14/25); C_r: mode 10 (17/25); alpha: mode 0.6 (7/25) |
| SPECTRE_global | LinearSVM | 0.544 +/- 0.053 | 0.837 +/- 0.028 | 0.707 +/- 0.031 | (ref) |  |  | C: mode 0.01 (17/25) |
| SPECTRE_regional_PCA64 | LinearSVM | 0.494 +/- 0.042 | 0.800 +/- 0.024 | 0.661 +/- 0.031 | -0.0499 | 0.0000*** | 0.0000*** | C: mode 0.1 (9/25) |
| Clean_Anat_PCA32_L2 | LinearSVM | 0.526 +/- 0.049 | 0.833 +/- 0.028 | 0.704 +/- 0.034 | -0.0184 | 0.0774 | 0.1135 | C: mode 0.01 (13/25) |
| Clean_Anat_PCA64_L2 | LinearSVM | 0.529 +/- 0.045 | 0.840 +/- 0.025 | 0.710 +/- 0.027 | -0.0153 | 0.0587 | 0.0710 | C: mode 0.01 (15/25) |
| Clean_Residual_PCA32_L2 | LinearSVM | 0.527 +/- 0.050 | 0.832 +/- 0.026 | 0.698 +/- 0.036 | -0.0173 | 0.0522 | 0.1135 | C: mode 0.01 (9/25) |
| Clean_Residual_PCA64_L2 | LinearSVM | 0.516 +/- 0.048 | 0.831 +/- 0.033 | 0.698 +/- 0.034 | -0.0287 | 0.0027** | 0.0025** | C: mode 0.01 (13/25) |
| Late_Fusion_Prob | LinearSVM | 0.544 +/- 0.051 | 0.841 +/- 0.025 | 0.712 +/- 0.031 | -0.0006 | 0.9145 | 0.6274 | C_g: mode 0.01 (17/25); C_r: mode 0.1 (9/25); alpha: mode 0.8 (5/25) |

### ROC-AUC and accuracy significance vs `SPECTRE_global` (B_tunedC)

| Config | Task | Model | dROC-AUC | t p | Wilcoxon p | dAccuracy | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| SPECTRE_regional_PCA64 | binary | LogReg | -0.0100 | 0.0008*** | 0.0006*** | -0.0200 | 0.0000*** | 0.0001*** |
| SPECTRE_regional_PCA64 | binary | LinearSVM | -0.0069 | 0.0019** | 0.0028** | -0.0202 | 0.0000*** | 0.0001*** |
| SPECTRE_regional_PCA64 | multiclass4 | LogReg | -0.0301 | 0.0000*** | 0.0000*** | -0.0526 | 0.0000*** | 0.0001*** |
| SPECTRE_regional_PCA64 | multiclass4 | LinearSVM | -0.0367 | 0.0000*** | 0.0000*** | -0.0458 | 0.0000*** | 0.0001*** |
| Clean_Anat_PCA32_L2 | binary | LogReg | -0.0000 | 0.9986 | 0.8949 | -0.0016 | 0.5521 | 0.8207 |
| Clean_Anat_PCA32_L2 | binary | LinearSVM | +0.0007 | 0.5888 | 0.6915 | -0.0002 | 0.9309 | 0.6796 |
| Clean_Anat_PCA32_L2 | multiclass4 | LogReg | -0.0102 | 0.0157* | 0.0342* | -0.0695 | 0.0000*** | 0.0000*** |
| Clean_Anat_PCA32_L2 | multiclass4 | LinearSVM | -0.0045 | 0.1885 | 0.2411 | -0.0027 | 0.6830 | 0.6069 |
| Clean_Anat_PCA64_L2 | binary | LogReg | -0.0015 | 0.2502 | 0.1409 | -0.0074 | 0.0109* | 0.0150* |
| Clean_Anat_PCA64_L2 | binary | LinearSVM | -0.0015 | 0.3047 | 0.3254 | -0.0090 | 0.0041** | 0.0052** |
| Clean_Anat_PCA64_L2 | multiclass4 | LogReg | -0.0042 | 0.3315 | 0.3388 | -0.0447 | 0.0000*** | 0.0001*** |
| Clean_Anat_PCA64_L2 | multiclass4 | LinearSVM | +0.0032 | 0.2745 | 0.2752 | +0.0027 | 0.6741 | 0.6995 |
| Clean_Residual_PCA32_L2 | binary | LogReg | -0.0026 | 0.0285* | 0.0347* | -0.0108 | 0.0000*** | 0.0002*** |
| Clean_Residual_PCA32_L2 | binary | LinearSVM | -0.0018 | 0.2424 | 0.3957 | -0.0103 | 0.0009*** | 0.0017** |
| Clean_Residual_PCA32_L2 | multiclass4 | LogReg | -0.0088 | 0.0252* | 0.0342* | -0.0744 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA32_L2 | multiclass4 | LinearSVM | -0.0047 | 0.0634 | 0.1266 | -0.0088 | 0.1893 | 0.1916 |
| Clean_Residual_PCA64_L2 | binary | LogReg | -0.0031 | 0.0213* | 0.0199* | -0.0139 | 0.0000*** | 0.0002*** |
| Clean_Residual_PCA64_L2 | binary | LinearSVM | -0.0029 | 0.0317* | 0.0370* | -0.0164 | 0.0000*** | 0.0001*** |
| Clean_Residual_PCA64_L2 | multiclass4 | LogReg | -0.0263 | 0.0000*** | 0.0000*** | -0.0726 | 0.0000*** | 0.0000*** |
| Clean_Residual_PCA64_L2 | multiclass4 | LinearSVM | -0.0062 | 0.1071 | 0.1135 | -0.0092 | 0.1479 | 0.1785 |
| Late_Fusion_Prob | binary | LogReg | +0.0053 | 0.0000*** | 0.0002*** | +0.0027 | 0.0917 | 0.0837 |
| Late_Fusion_Prob | binary | LinearSVM | +0.0074 | 0.0000*** | 0.0000*** | +0.0050 | 0.0045** | 0.0083** |
| Late_Fusion_Prob | multiclass4 | LogReg | +0.0055 | 0.0302* | 0.0345* | +0.0130 | 0.0008*** | 0.0010** |
| Late_Fusion_Prob | multiclass4 | LinearSVM | +0.0040 | 0.0011** | 0.0017** | +0.0054 | 0.0688 | 0.0787 |

## Best configuration per setting (by mean macro-F1)

| Regime | Task | Model | Best config | Macro-F1 | Baseline Macro-F1 | dMacro-F1 | t p | Wilcoxon p |
|---|---|---|---|---|---|---|---|---|
| A_fixed | binary | LogReg | Late_Fusion_Prob | 0.862 | 0.861 | +0.0009 | 0.5857 | 0.5677 |
| A_fixed | binary | LinearSVM | SPECTRE_global | 0.853 | 0.853 | +0.0000 | - | - |
| A_fixed | multiclass4 | LogReg | Late_Fusion_Prob | 0.523 | 0.518 | +0.0045 | 0.4151 | 0.3173 |
| A_fixed | multiclass4 | LinearSVM | SPECTRE_global | 0.551 | 0.551 | +0.0000 | - | - |
| B_tunedC | binary | LogReg | Late_Fusion_Prob | 0.862 | 0.858 | +0.0038 | 0.0764 | 0.0944 |
| B_tunedC | binary | LinearSVM | Late_Fusion_Prob | 0.864 | 0.858 | +0.0068 | 0.0020** | 0.0030** |
| B_tunedC | multiclass4 | LogReg | Late_Fusion_Prob | 0.523 | 0.516 | +0.0069 | 0.1734 | 0.0765 |
| B_tunedC | multiclass4 | LinearSVM | SPECTRE_global | 0.544 | 0.544 | +0.0000 | - | - |

Significance markers: p < 0.05 `*`, p < 0.01 `**`, p < 0.001 `***`. Differences are variant - SPECTRE_global (positive = variant better). No correction for multiple comparisons is applied; with 6 variants x 2 tasks x 2 models x 2 regimes = 48 comparisons per metric, isolated p-values near 0.05 should be read with caution.
