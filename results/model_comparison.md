# Model Comparison: CT-FM vs SPECTRE vs TAP-CT on MosMedData

- **Dataset:** MosMedData, 1,110 chest CT scans (CT-0: 254, CT-1: 684, CT-2: 125, CT-3: 45, CT-4: 2); native spacing ~0.74×0.74×8 mm
- **Protocol:** frozen encoder, one embedding per scan, then Linear Probing with 5-fold Stratified CV (repeated 5×, mean ± std over 25 folds). StandardScaler is fit inside each training fold; `class_weight="balanced"`; LogReg C=1, Linear SVM C=0.01
- **Per-model details:** [`ct_fm_results.md`](ct_fm_results.md), [`spectre_results.md`](spectre_results.md), [`tap_ct_results.md`](tap_ct_results.md)

## Models

| | CT-FM | SPECTRE | TAP-CT |
|---|---|---|---|
| Checkpoint | `project-lighter/ct_fm_feature_extractor` | SPECTRE-large (`cclaess/SPECTRE`) | `fomofo/tap-ct-b-3d` |
| Architecture | SegResNet encoder (CNN) | ViT-L crop encoder + ViT feature combiner | 3D ViT-B (DINOv2-style) |
| Pretraining | SimCLR (self-supervised) | SSL + vision-language alignment | DINOv2-style SSL |
| Preprocessing (official) | SPL, 3×1×1 mm, foreground crop, HU [-1024, 2048] → [0, 1] | RAS, 0.5×0.5×1 mm, 256 mm center cube, HU [-1000, 1000] → [0, 1], 64 crops | LPS, in-plane 224×224, native slices, HU [-1008, 822], z-score |
| Embedding | Global average pool, 512-d | Combiner [CLS], 1080-d | [CLS], 768-d |

## Results

Bold is the best model in each row.

| Task | Classifier | Metric | CT-FM | SPECTRE | TAP-CT |
|---|---|---|---|---|---|
| **Binary** (CT-0 vs CT-1..4) | LogReg | Accuracy | 0.710 ± 0.030 | **0.893 ± 0.020** | 0.723 ± 0.025 |
| | | Macro-F1 | 0.633 ± 0.028 | **0.861 ± 0.023** | 0.622 ± 0.032 |
| | | ROC-AUC | 0.722 ± 0.031 | **0.946 ± 0.016** | 0.695 ± 0.040 |
| | Linear SVM | Accuracy | 0.688 ± 0.031 | **0.881 ± 0.022** | 0.705 ± 0.023 |
| | | Macro-F1 | 0.625 ± 0.029 | **0.852 ± 0.024** | 0.620 ± 0.028 |
| | | ROC-AUC | 0.723 ± 0.032 | **0.949 ± 0.015** | 0.699 ± 0.038 |
| **Multiclass, 5 classes** (CT-0..CT-4) | LogReg | Accuracy | 0.467 ± 0.027 | **0.655 ± 0.029** | 0.519 ± 0.024 |
| | | Macro-F1 | 0.271 ± 0.023 | **0.410 ± 0.030** | 0.272 ± 0.027 |
| | | ROC-AUC | 0.595 ± 0.058 | **0.802 ± 0.037** | 0.622 ± 0.039 |
| | Linear SVM | Accuracy | 0.541 ± 0.029 | **0.710 ± 0.025** | 0.509 ± 0.033 |
| | | Macro-F1 | 0.290 ± 0.023 | **0.433 ± 0.036** | 0.271 ± 0.030 |
| | | ROC-AUC | 0.635 ± 0.039 | **0.840 ± 0.028** | 0.591 ± 0.040 |
| **Multiclass, 4 classes** (CT-3 and CT-4 merged) | LogReg | Accuracy | 0.469 ± 0.024 | **0.661 ± 0.031** | 0.519 ± 0.026 |
| | | Macro-F1 | 0.346 ± 0.035 | **0.519 ± 0.045** | 0.340 ± 0.032 |
| | | ROC-AUC | 0.630 ± 0.035 | **0.820 ± 0.027** | 0.613 ± 0.032 |
| | Linear SVM | Accuracy | 0.539 ± 0.027 | **0.713 ± 0.028** | 0.511 ± 0.026 |
| | | Macro-F1 | 0.372 ± 0.037 | **0.551 ± 0.052** | 0.340 ± 0.027 |
| | | ROC-AUC | 0.654 ± 0.030 | **0.839 ± 0.028** | 0.605 ± 0.032 |

Multiclass ROC-AUC is the macro average of one-vs-rest AUCs. CT-4 has only 2 scans, so most test folds don't contain it; the 4-class task (CT-3 and CT-4 merged) is the more reliable multiclass result.

## Key takeaways

1. **SPECTRE is best in all 18 rows.** It leads by +0.18 to +0.25 ROC-AUC and +0.14 to +0.23 Macro-F1 over the other two models, while the fold-to-fold std is about 0.03. Binary ROC-AUC reaches **0.95**.
2. **CT-FM and TAP-CT are roughly tied.** Most of their differences are smaller than one std: TAP-CT is slightly ahead on LogReg accuracy, CT-FM slightly ahead on ROC-AUC and on the SVM rows.
3. **Severity grading is hard for every model.** Even SPECTRE reaches only 0.43 Macro-F1 on 5 classes and 0.55 on 4 classes. Most errors are between neighbouring grades, especially CT-0 vs CT-1 (less than 25% of the lung affected).
4. **Use Macro-F1 and ROC-AUC rather than accuracy.** With balanced class weights, CT-FM's and TAP-CT's binary accuracy (about 0.70–0.72) is below the 0.77 majority-class baseline; only SPECTRE beats it (0.89).
5. **This compares each model with its official preprocessing, not the encoders in isolation.** Likely contributors, not tested:
   - SPECTRE resamples to fine voxels and merges 64 crops with a learned combiner; it was also pretrained with radiology reports.
   - TAP-CT keeps MosMed's 8 mm slices (only about 11 patches deep), while it was pretrained on 12-slice crops.
   - CT-FM averages its features over the whole body crop.
6. **Possible next steps:** lung masking or cropping before pooling; resampling slices for TAP-CT; CLS + mean-patch pooling; nested-CV tuning of C; fine-tuning SPECTRE instead of a linear probe.

## Reproduce

```bash
python src/extract_ct_fm.py   --out outputs/ct_fm_embeddings.npz     # CPU ~3.5 h (GPU auto-used if free)
python src/extract_spectre.py --out outputs/spectre_embeddings.npz   # GPU ~17 min
python src/extract_tap_ct.py  --out outputs/tap_ct_embeddings.npz    # GPU ~4 min
for m in ct_fm spectre tap_ct; do
  python src/evaluate_embeddings.py outputs/${m}_embeddings.npz --repeats 5 --json outputs/${m}_eval.json
done
```
