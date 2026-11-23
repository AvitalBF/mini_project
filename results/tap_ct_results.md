# TAP-CT Baseline Evaluation Results

- **Model:** TAP-CT-B-3D (`fomofo/tap-ct-b-3d`, revision `7694ac2`), frozen 3D ViT-Base (DINOv2-style), 768-d final [CLS] token (`pooler_output`) from a single whole-volume forward pass
- **Dataset:** MosMedData, 1,110 scans (CT-0: 254, CT-1: 684, CT-2: 125, CT-3: 45, CT-4: 2)
- **Preprocessing:** TAP-CT's official image processor: LPS orientation, in-plane resize to 224×224 (native slices kept), z-padding to a multiple of 4 with -1024 HU, HU clip [-1008, 822], z-score (mean -86.81, std 322.63). Inference in fp32; attention computed with PyTorch SDPA (numerically identical to the model's attention fallback, cos = 1.0)
- **Evaluation:** 5-fold Stratified CV (repeated 5×, mean ± std over 25 folds), Linear Probing (StandardScaler fit inside each training fold; `class_weight="balanced"`; LogReg C=1, Linear SVM C=0.01)

| Task | Model | Accuracy | Macro-F1 | ROC-AUC |
|---|---|---|---|---|
| **Binary** (CT-0 vs CT-1..4) | LogReg | **0.723 ± 0.025** | **0.622 ± 0.032** | 0.695 ± 0.040 |
| | Linear SVM | 0.705 ± 0.023 | 0.620 ± 0.028 | **0.699 ± 0.038** |
| **Multiclass, 5 classes** (CT-0..CT-4) | LogReg | **0.519 ± 0.024** | **0.272 ± 0.027** | **0.622 ± 0.039** |
| | Linear SVM | 0.509 ± 0.033 | 0.271 ± 0.030 | 0.591 ± 0.040 |
| **Multiclass, 4 classes** (CT-3 and CT-4 merged) | LogReg | **0.519 ± 0.026** | 0.340 ± 0.032 | **0.613 ± 0.032** |
| | Linear SVM | 0.511 ± 0.026 | 0.340 ± 0.027 | 0.605 ± 0.032 |

Multiclass ROC-AUC is the macro average of one-vs-rest AUCs. CT-4 has only 2 scans, so most test folds don't contain it.

Reproduce:
```bash
python src/extract_tap_ct.py --out outputs/tap_ct_embeddings.npz
python src/evaluate_embeddings.py outputs/tap_ct_embeddings.npz --repeats 5 --json outputs/tap_ct_eval.json
```
