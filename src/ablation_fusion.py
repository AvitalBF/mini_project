#!/usr/bin/env python
"""Controlled ablation of SPECTRE/SAMv2 fusion variants on outputs/spectre_samv2_fusion.npz,
using the *exact* evaluation protocol of evaluate_embeddings.py (same imported functions, same
classifiers/hyperparameters, same 5-fold x 5-repeat stratified CV with seeds 42..46 -> 25 folds),
so every feature configuration below is scored on identical train/test splits per task. That
makes paired significance tests across configurations valid (same y, same seed -> StratifiedKFold
produces identical fold membership regardless of what X looks like).

Feature configurations (dims)
  SPECTRE_global           1080  extract_spectre.py's whole-scan embedding; baseline reproduction.
  SAMv2_only               1176  samv2_global_pooled(384) + zone_embeddings(6x128=768)
                                 + zone_scores(6) + zone_coords_mm(6x3=18).
  Global_fusion            1464  spectre_global + samv2_global_pooled (no anatomical alignment).
  Anatomical_fusion_raw    8736  spectre_global + spectre_regional(6x1080=6480) + SAMv2_only.
  Anatomical_fusion_PCA64  2384  spectre_global + SAMv2_only + 64-d PCA(spectre_regional flat)
                                 + 64-d PCA(regional - global residual flat).
  Anatomical_fusion_PCA128 2512  same, with 128 components each.
  The PCA configs fit PCA *inside* each training fold only (sklearn ColumnTransformer inside the
  cross-validated Pipeline), so there is no leakage from the held-out fold into the reduction.

Outputs: outputs/ablation_results.json (full per-fold metrics + paired tests vs SPECTRE_global)
and outputs/ablation_summary.md (formatted report).
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel, wilcoxon
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluate_embeddings import make_models, make_tasks, metrics, scores_full

FUSION_NPZ_DEFAULT = Path("/workspace/outputs/spectre_samv2_fusion.npz")
FOLDS, REPEATS, SEED = 5, 5, 42  # matches evaluate_embeddings.py defaults: 5x5=25 folds, seeds 42..46
C_LR, C_SVM = 1.0, 0.01
MODEL_NAMES = ["LogReg", "LinearSVM"]
TASKS = ["binary", "multiclass4", "multiclass"]
PCA_COMPONENTS = [64, 128]
BASELINE = "SPECTRE_global"
# multiclass (5-class) + "balanced" class weighting makes lbfgs converge pathologically slowly
# when CT-4 has only 2 samples (extreme per-sample weight); skipped for the heaviest config only.
SKIP_TASK_FOR_CONFIG = {("Anatomical_fusion_raw", "multiclass")}


# --------------------------------------------------------------------------------- feature sets
def samv2_only(d: dict, n: int) -> np.ndarray:
    return np.concatenate([
        d["samv2_global_pooled"],
        d["samv2_zone_embeddings"].reshape(n, -1),
        d["samv2_zone_scores"],
        d["samv2_zone_coords_mm"].reshape(n, -1),
    ], axis=1).astype(np.float64)


def build_feature_sets(d: dict) -> dict:
    n = len(d["labels"])
    spectre_global = d["spectre_global"].astype(np.float64)
    samv2 = samv2_only(d, n)
    regional_flat = d["spectre_regional"].reshape(n, -1).astype(np.float64)
    residual_flat = (d["spectre_regional"] - d["spectre_global"][:, None, :]).reshape(n, -1).astype(np.float64)
    passthrough = np.concatenate([spectre_global, samv2], axis=1)  # 1080 + 1176 = 2256

    pca_raw = np.concatenate([passthrough, regional_flat, residual_flat], axis=1)
    pca_slices = dict(
        passthrough=slice(0, passthrough.shape[1]),
        regional=slice(passthrough.shape[1], passthrough.shape[1] + regional_flat.shape[1]),
        residual=slice(passthrough.shape[1] + regional_flat.shape[1], pca_raw.shape[1]),
    )

    sets = {
        "SPECTRE_global": dict(X=spectre_global, n_features=spectre_global.shape[1]),
        "SAMv2_only": dict(X=samv2, n_features=samv2.shape[1]),
        "Global_fusion": dict(X=np.concatenate([spectre_global, d["samv2_global_pooled"]], axis=1),
                               n_features=spectre_global.shape[1] + d["samv2_global_pooled"].shape[1]),
        "Anatomical_fusion_raw": dict(X=np.concatenate([spectre_global, regional_flat, samv2], axis=1),
                                       n_features=spectre_global.shape[1] + regional_flat.shape[1] + samv2.shape[1]),
    }
    for k in PCA_COMPONENTS:
        sets[f"Anatomical_fusion_PCA{k}"] = dict(X=pca_raw, pca_k=k, pca_slices=pca_slices,
                                                   n_features=passthrough.shape[1] + 2 * k)
    return sets


def base_classifier(name: str, seed: int):
    """The unfitted classifier step from evaluate_embeddings.make_models, so hyperparameters
    (C, class_weight, max_iter) can never drift from the baseline script's."""
    return make_models(C_LR, C_SVM, seed)[name].steps[-1][1]


def build_proto(feature_set: dict, model_name: str, seed: int):
    if "pca_k" not in feature_set:
        return make_models(C_LR, C_SVM, seed)[model_name]
    k, slices = feature_set["pca_k"], feature_set["pca_slices"]
    reduce = ColumnTransformer([
        ("passthrough", "passthrough", slices["passthrough"]),
        ("pca_regional", PCA(n_components=k, random_state=42), slices["regional"]),
        ("pca_residual", PCA(n_components=k, random_state=42), slices["residual"]),
    ])
    return Pipeline([("reduce", reduce), ("scale", StandardScaler()), ("clf", base_classifier(model_name, seed))])


# --------------------------------------------------------------------------------- evaluation
def evaluate_detailed(X: np.ndarray, y: np.ndarray, n_classes: int, proto, folds: int, repeats: int, seed: int) -> dict:
    """Same loop as evaluate_embeddings.evaluate(), plus raw per-fold metric arrays (length
    folds*repeats) for paired significance testing across feature configurations."""
    fold_rows, pooled_rows, cms = [], [], []
    per_fold = {"accuracy": [], "macro_f1": [], "roc_auc": []}
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        oof_pred = np.empty(len(y), dtype=int)
        oof_s = np.empty(len(y)) if n_classes == 2 else np.empty((len(y), n_classes))
        for tr, te in skf.split(X, y):
            m = clone(proto).fit(X[tr], y[tr])
            pred, s = m.predict(X[te]), scores_full(m, X[te], n_classes)
            oof_pred[te], oof_s[te] = pred, s
            fm = metrics(y[te], pred, s, n_classes)
            fold_rows.append(fm)
            for k in per_fold:
                per_fold[k].append(fm[k])
        pooled_rows.append(metrics(y, oof_pred, oof_s, n_classes))
        cms.append(confusion_matrix(y, oof_pred, labels=np.arange(n_classes)))

    def agg(rows):
        return {k: [float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))] for k in rows[0]}

    return dict(fold=agg(fold_rows), pooled=agg(pooled_rows), confusion=np.mean(cms, axis=0).tolist(),
                per_fold=per_fold)


def paired_tests(a: list, b: list) -> dict:
    """Paired t-test + Wilcoxon signed-rank on matched per-fold values (same CV splits), with
    NaN folds (e.g. a class absent from a multiclass test fold) dropped pairwise."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    out = dict(n=int(mask.sum()), mean_diff=float(np.mean(a - b)) if mask.sum() else None)
    if mask.sum() < 3 or np.allclose(a, b):
        out.update(t_stat=None, t_p=None, wilcoxon_stat=None, wilcoxon_p=None)
        return out
    t_stat, t_p = ttest_rel(a, b)
    try:
        w_stat, w_p = wilcoxon(a, b)
    except ValueError:  # all differences zero after rounding, etc.
        w_stat, w_p = None, None
    out.update(t_stat=float(t_stat), t_p=float(t_p),
               wilcoxon_stat=float(w_stat) if w_stat is not None else None,
               wilcoxon_p=float(w_p) if w_p is not None else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="?", default=FUSION_NPZ_DEFAULT)
    ap.add_argument("--json-out", type=Path, default=Path("/workspace/outputs/ablation_results.json"))
    ap.add_argument("--md-out", type=Path, default=Path("/workspace/outputs/ablation_summary.md"))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    labels = d["labels"].astype(int)
    n = len(labels)
    print(f"{args.npz}: {n} scans, class counts CT-0..4: {np.bincount(labels, minlength=5).tolist()}")

    feature_sets = build_feature_sets(d)
    tasks = make_tasks(labels)
    for name, fs in feature_sets.items():
        print(f"[config] {name:<26} n_features={fs['n_features']}")

    results: dict = {}
    t_start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        warnings.filterwarnings("ignore", message=".*least populated class.*")
        for config_name, fs in feature_sets.items():
            results[config_name] = {}
            for task in TASKS:
                if (config_name, task) in SKIP_TASK_FOR_CONFIG:
                    print(f"[skip] {config_name} / {task}: 5-class LogReg is pathologically slow "
                          f"here (class_weight=balanced + CT-4 n=2); skipped by request", flush=True)
                    continue
                y, class_names = tasks[task]
                n_classes = len(class_names)
                counts = np.bincount(y, minlength=n_classes)
                if (counts > 0).sum() < 2:
                    continue
                results[config_name][task] = {}
                for model_name in MODEL_NAMES:
                    t0 = time.perf_counter()
                    proto = build_proto(fs, model_name, SEED)
                    res = evaluate_detailed(fs["X"], y, n_classes, proto, FOLDS, REPEATS, SEED)
                    results[config_name][task][model_name] = res
                    dt = time.perf_counter() - t0
                    mf1 = res["fold"]["macro_f1"]
                    print(f"[{config_name:<26}] {task:<11} {model_name:<10} "
                          f"macro_f1={mf1[0]:.3f}+/-{mf1[1]:.3f} ({dt:.1f}s)", flush=True)

    print(f"[eval] total {time.perf_counter()-t_start:.1f}s", flush=True)

    # ------------------------------------------------------------------- significance vs baseline
    significance: dict = {}
    for config_name in feature_sets:
        if config_name == BASELINE:
            continue
        significance[config_name] = {}
        for task in TASKS:
            if task not in results.get(config_name, {}) or task not in results.get(BASELINE, {}):
                continue
            significance[config_name][task] = {}
            for model_name in MODEL_NAMES:
                a = results[config_name][task][model_name]["per_fold"]
                b = results[BASELINE][task][model_name]["per_fold"]
                significance[config_name][task][model_name] = {
                    "macro_f1": paired_tests(a["macro_f1"], b["macro_f1"]),
                    "roc_auc": paired_tests(a["roc_auc"], b["roc_auc"]),
                }

    payload = dict(
        protocol=dict(folds=FOLDS, repeats=REPEATS, seed=SEED, seeds=list(range(SEED, SEED + REPEATS)),
                      total_folds=FOLDS * REPEATS, C_lr=C_LR, C_svm=C_SVM, models=MODEL_NAMES, tasks=TASKS,
                      baseline=BASELINE, source_npz=str(args.npz)),
        configs={k: {"n_features": v["n_features"]} for k, v in feature_sets.items()},
        results=results,
        significance_vs_baseline=significance,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {args.json_out}")

    write_markdown_report(args.md_out, feature_sets, results, significance)
    print(f"[saved] {args.md_out}")


# --------------------------------------------------------------------------------- report
def write_markdown_report(path: Path, feature_sets: dict, results: dict, significance: dict) -> None:
    def fmt(mean_std):
        return f"{mean_std[0]:.3f} +/- {mean_std[1]:.3f}"

    def p_str(p):
        if p is None:
            return "n/a"
        stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        return f"{p:.4f}{stars}"

    lines = []
    lines.append("# SPECTRE / SAMv2 Fusion Ablation\n")
    lines.append(f"Protocol: {FOLDS}-fold StratifiedKFold x {REPEATS} repeats "
                 f"(seeds {SEED}..{SEED + REPEATS - 1}, {FOLDS * REPEATS} folds total), "
                 f"StandardScaler fit strictly inside each training fold, "
                 f"LogReg (C={C_LR}, class_weight=balanced, max_iter=5000) and "
                 f"LinearSVM (C={C_SVM}, class_weight=balanced, max_iter=20000) "
                 f"-- identical to evaluate_embeddings.py. PCA (where used) is fit inside each "
                 f"training fold only.\n")
    if SKIP_TASK_FOR_CONFIG:
        skipped = ", ".join(f"{c} / {t}" for c, t in sorted(SKIP_TASK_FOR_CONFIG))
        lines.append(f"**Skipped:** {skipped} -- the 5-class `multiclass` task combined with "
                     f"`class_weight=\"balanced\"` (CT-4 has only 2 samples, so lbfgs sees an "
                     f"extreme per-sample weight) made LogReg converge pathologically slowly on "
                     f"this config's 8736 raw dimensions; `binary` and `multiclass4` still ran "
                     f"with the exact, unmodified protocol for every configuration.\n")

    lines.append("## Feature configurations\n")
    lines.append("| Config | Dims |")
    lines.append("|---|---|")
    for name, fs in feature_sets.items():
        lines.append(f"| {name} | {fs['n_features']} |")
    lines.append("")

    lines.append("## Full comparison (fold-level mean +/- std over 25 folds)\n")
    lines.append("| Config | Task | Model | Macro-F1 | ROC-AUC | Accuracy |")
    lines.append("|---|---|---|---|---|---|")
    for config_name in feature_sets:
        for task in TASKS:
            if task not in results.get(config_name, {}):
                continue
            for model_name in MODEL_NAMES:
                r = results[config_name][task][model_name]["fold"]
                lines.append(f"| {config_name} | {task} | {model_name} | "
                             f"{fmt(r['macro_f1'])} | {fmt(r['roc_auc'])} | {fmt(r['accuracy'])} |")
    lines.append("")

    lines.append(f"## Significance vs {BASELINE} (paired t-test / Wilcoxon on the matched 25 folds)\n")
    lines.append("### Macro-F1, task=multiclass4 (highlighted: this is the main severity-ordinal task)\n")
    lines.append("| Config | Model | Mean diff | t_p | wilcoxon_p | n folds |")
    lines.append("|---|---|---|---|---|---|")
    for config_name, by_task in significance.items():
        if "multiclass4" not in by_task:
            continue
        for model_name in MODEL_NAMES:
            s = by_task["multiclass4"][model_name]["macro_f1"]
            md = f"{s['mean_diff']:+.4f}" if s["mean_diff"] is not None else "n/a"
            lines.append(f"| {config_name} | {model_name} | {md} | {p_str(s['t_p'])} | {p_str(s['wilcoxon_p'])} | {s['n']} |")
    lines.append("")

    lines.append("### ROC-AUC, task=binary\n")
    lines.append("| Config | Model | Mean diff | t_p | wilcoxon_p | n folds |")
    lines.append("|---|---|---|---|---|---|")
    for config_name, by_task in significance.items():
        if "binary" not in by_task:
            continue
        for model_name in MODEL_NAMES:
            s = by_task["binary"][model_name]["roc_auc"]
            md = f"{s['mean_diff']:+.4f}" if s["mean_diff"] is not None else "n/a"
            lines.append(f"| {config_name} | {model_name} | {md} | {p_str(s['t_p'])} | {p_str(s['wilcoxon_p'])} | {s['n']} |")
    lines.append("")

    lines.append("### All tasks, Macro-F1 (full significance table)\n")
    lines.append("| Config | Task | Model | Mean diff | t_p | wilcoxon_p | n folds |")
    lines.append("|---|---|---|---|---|---|---|")
    for config_name, by_task in significance.items():
        for task, by_model in by_task.items():
            for model_name in MODEL_NAMES:
                s = by_model[model_name]["macro_f1"]
                md = f"{s['mean_diff']:+.4f}" if s["mean_diff"] is not None else "n/a"
                lines.append(f"| {config_name} | {task} | {model_name} | {md} | {p_str(s['t_p'])} | "
                             f"{p_str(s['wilcoxon_p'])} | {s['n']} |")
    lines.append("")
    lines.append("p < 0.05 `*`, p < 0.01 `**`, p < 0.001 `***`. Mean diff = config - SPECTRE_global "
                 "(positive = fusion variant better).\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
