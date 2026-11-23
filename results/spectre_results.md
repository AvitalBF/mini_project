# SPECTRE Baseline Evaluation Results

- **Model:** SPECTRE-large (`spectre-fm`, weights from `cclaess/SPECTRE`), frozen ViT-L crop encoder + feature combiner, 1080-d combiner CLS-token embedding
- **Dataset:** MosMedData, 1,110 scans (CT-0: 254, CT-1: 684, CT-2: 125, CT-3: 45, CT-4: 2)
- **Preprocessing:** SPECTRE's official linear-probing pipeline: HU clip [-1000, 1000] scaled to [0, 1], RAS orientation, resample to 0.5×0.5×1.0 mm, center pad/crop to 512×512×256, 4×4×4 grid of 128×128×64 crops
- **Evaluation:** 5-fold Stratified CV (repeated 5×, mean ± std over 25 folds), Linear Probing (StandardScaler fit inside each training fold; `class_weight="balanced"`; LogReg C=1, Linear SVM C=0.01)

| Task | Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| **Binary** (CT-0 vs CT-1..4) | LogReg | **0.893 ± 0.020** | **0.861 ± 0.023** | 0.946 ± 0.016 |
| | Linear SVM | 0.881 ± 0.022 | 0.852 ± 0.024 | **0.949 ± 0.015** |
| **Multiclass, 5 classes** (CT-0..CT-4) | LogReg | 0.655 ± 0.029 | 0.410 ± 0.030 | 0.802 ± 0.037 |
| | Linear SVM | **0.710 ± 0.025** | **0.433 ± 0.036** | **0.840 ± 0.028** |
| **Multiclass, 4 classes** (CT-3 and CT-4 merged) | LogReg | 0.661 ± 0.031 | 0.519 ± 0.045 | 0.820 ± 0.027 |
| | Linear SVM | **0.713 ± 0.028** | **0.551 ± 0.052** | **0.839 ± 0.028** |

Multiclass ROC-AUC is the macro average of one-vs-rest AUCs. CT-4 has only 2 scans, so most test folds don't contain it.

Reproduce:
```bash
python src/extract_spectre.py --out outputs/spectre_embeddings.npz
python src/evaluate_embeddings.py outputs/spectre_embeddings.npz --repeats 5 --json outputs/spectre_eval.json
```
