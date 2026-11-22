# CT-FM Baseline Evaluation Results

- **Model:** CT-FM (`project-lighter/ct_fm_feature_extractor`), frozen encoder, 512-d global-average-pooled embedding
- **Dataset:** MosMedData, 1,110 scans (CT-0: 254, CT-1: 684, CT-2: 125, CT-3: 45, CT-4: 2)
- **Preprocessing:** CT-FM pretraining pipeline: Orientation SPL, resample to 3×1×1 mm, foreground crop, HU clip [-1024, 2048] scaled to [0, 1]
- **Evaluation:** 5-fold Stratified CV (repeated 5×, mean ± std over 25 folds), Linear Probing (StandardScaler fit inside each training fold; `class_weight="balanced"`; LogReg C=1, Linear SVM C=0.01)

| Task | Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| **Binary** (CT-0 vs CT-1..4) | LogReg | **0.710 ± 0.030** | **0.633 ± 0.028** | 0.722 ± 0.031 |
| | Linear SVM | 0.688 ± 0.031 | 0.625 ± 0.029 | **0.723 ± 0.032** |
| **Multiclass, 5 classes** (CT-0..CT-4) | LogReg | 0.467 ± 0.027 | 0.271 ± 0.023 | 0.595 ± 0.058 |
| | Linear SVM | **0.541 ± 0.029** | **0.290 ± 0.023** | **0.635 ± 0.039** |
| **Multiclass, 4 classes** (CT-3 and CT-4 merged) | LogReg | 0.469 ± 0.024 | 0.346 ± 0.035 | 0.630 ± 0.035 |
| | Linear SVM | **0.539 ± 0.027** | **0.372 ± 0.037** | **0.654 ± 0.030** |

Multiclass ROC-AUC is the macro average of one-vs-rest AUCs. CT-4 has only 2 scans, so most test folds don't contain it.

Reproduce:
```bash
python src/extract_ct_fm.py --out outputs/ct_fm_embeddings.npz
python src/evaluate_embeddings.py outputs/ct_fm_embeddings.npz --repeats 5 --json outputs/ct_fm_eval.json
```
