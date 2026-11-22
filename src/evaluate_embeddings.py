#!/usr/bin/env python
"""Linear-probe evaluation of precomputed scan embeddings (.npz from extract_*.py).

Tasks
  binary      CT-0 (normal) vs CT-1..CT-4 (abnormal)
  multiclass  CT-0..CT-4 (5 classes; CT-4 has only 2 scans in MosMedData)
  multiclass4 CT-0, CT-1, CT-2, CT-3+4 (merged; reported alongside since CT-4 is too small)

Protocol
  Stratified K-fold CV (default 5, repeated --repeats times with different seeds).
  StandardScaler is inside the Pipeline, so it is fit on the training folds only (no leakage).
  Models: LogisticRegression and LinearSVC, both class_weight="balanced".
  Metrics per fold: accuracy, macro-F1, ROC-AUC (binary: AUC; multiclass: macro one-vs-rest AUC
  over classes present in the test fold). Reported as mean +/- std over folds, plus "pooled"
  metrics computed once on all out-of-fold predictions (more stable for the tiny classes).
  LinearSVC AUC uses decision_function scores (AUC is rank-based, no probabilities needed).
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def make_models(C_lr: float, C_svm: float, seed: int) -> dict:
    return {
        "LogReg": make_pipeline(
            StandardScaler(), LogisticRegression(C=C_lr, class_weight="balanced", max_iter=5000, random_state=seed)
        ),
        "LinearSVM": make_pipeline(
            StandardScaler(), LinearSVC(C=C_svm, class_weight="balanced", max_iter=20000, random_state=seed)
        ),
    }


def make_tasks(labels: np.ndarray) -> dict:
    return {
        "binary": ((labels > 0).astype(int), ["CT-0", "CT-1..4"]),
        "multiclass": (labels.copy(), ["CT-0", "CT-1", "CT-2", "CT-3", "CT-4"]),
        "multiclass4": (np.minimum(labels, 3), ["CT-0", "CT-1", "CT-2", "CT-3+4"]),
    }


def scores_full(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Continuous scores of shape (n, n_classes) (or (n,) for binary), aligned to label ids 0..n_classes-1.
    Classes missing from a training fold get -inf (never ranked as that class)."""
    s = model.predict_proba(X) if hasattr(model, "predict_proba") else model.decision_function(X)
    cls = model.classes_
    if n_classes == 2:
        return s[:, 1] if s.ndim == 2 else s
    if s.ndim == 1:  # only 2 classes seen in training
        s = np.stack([-s, s], 1)
    out = np.full((len(X), n_classes), -np.inf)
    out[:, cls] = s
    return out


def auc(y: np.ndarray, s: np.ndarray, n_classes: int) -> float:
    if n_classes == 2:
        return roc_auc_score(y, s) if len(np.unique(y)) == 2 else np.nan
    per = []
    for c in range(n_classes):
        pos = y == c
        if 0 < pos.sum() < len(y):
            col = np.nan_to_num(s[:, c], neginf=-1e12)
            per.append(roc_auc_score(pos, col))
    return float(np.mean(per)) if per else np.nan


def metrics(y, pred, s, n_classes) -> dict:
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", labels=np.arange(n_classes), zero_division=0),
        "roc_auc": auc(y, s, n_classes),
    }


def evaluate(X, y, n_classes, model_proto, folds, repeats, seed):
    fold_rows, pooled_rows, cms = [], [], []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed + r)
        oof_pred = np.empty(len(y), dtype=int)
        oof_s = np.empty(len(y)) if n_classes == 2 else np.empty((len(y), n_classes))
        for tr, te in skf.split(X, y):
            m = clone(model_proto).fit(X[tr], y[tr])
            pred, s = m.predict(X[te]), scores_full(m, X[te], n_classes)
            oof_pred[te], oof_s[te] = pred, s
            fold_rows.append(metrics(y[te], pred, s, n_classes))
        pooled_rows.append(metrics(y, oof_pred, oof_s, n_classes))
        cms.append(confusion_matrix(y, oof_pred, labels=np.arange(n_classes)))

    def agg(rows):
        return {k: (float(np.nanmean([r[k] for r in rows])), float(np.nanstd([r[k] for r in rows]))) for k in rows[0]}

    return agg(fold_rows), agg(pooled_rows), np.mean(cms, 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, help="embeddings file with arrays `embeddings` and `labels`")
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass", "multiclass4"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=1, help="repeat CV with different shuffles")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--C-lr", type=float, default=1.0)
    ap.add_argument("--C-svm", type=float, default=0.01)
    ap.add_argument("--json", type=Path, default=None, help="optional path to save results as JSON")
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    X, labels = d["embeddings"].astype(np.float64), d["labels"].astype(int)
    print(f"{args.npz}: X={X.shape}, model={d['model_id'] if 'model_id' in d else '?'}, "
          f"prep={d['preprocessing'] if 'preprocessing' in d else '?'}")
    print(f"class counts CT-0..4: {np.bincount(labels, minlength=5).tolist()}")
    print(f"{args.folds}-fold stratified CV x{args.repeats}; StandardScaler fit inside each training fold\n")

    results = {}
    tasks = make_tasks(labels)
    for task in args.tasks:
        y, names = tasks[task]
        n_classes = len(names)
        counts = np.bincount(y, minlength=n_classes)
        present = counts > 0
        if present.sum() < 2:
            print(f"== {task}: skipped (only one class present)\n")
            continue
        if counts[present].min() < args.folds:
            print(f"!! {task}: smallest class has {counts[present].min()} samples < {args.folds} folds; "
                  "some test folds will lack it (per-fold AUC averages over classes present).")
        print(f"== {task} ({' / '.join(f'{n}:{c}' for n, c in zip(names, counts))})")
        print(f"   {'model':<10} {'':>7} {'accuracy':>15} {'macro_F1':>15} {'ROC_AUC':>15}")
        results[task] = {}
        for name, proto in make_models(args.C_lr, args.C_svm, args.seed).items():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                warnings.filterwarnings("ignore", message=".*least populated class.*")
                fold, pooled, cm = evaluate(X, y, n_classes, proto, args.folds, args.repeats, args.seed)
            for tag, res in (("fold", fold), ("pooled", pooled)):
                cells = " ".join(f"{m:>7.3f} ± {s:<5.3f}" for m, s in res.values())
                print(f"   {name:<10} {tag:>7} {cells}")
            print("   confusion (rows=true, cols=pred, pooled OOF):\n" +
                  "\n".join("     " + f"{names[i]:>8} " + " ".join(f"{v:6.1f}" for v in row) for i, row in enumerate(cm)))
            results[task][name] = {"fold": fold, "pooled": pooled, "confusion": cm.tolist()}
        print()

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2))
        print(f"saved -> {args.json}")


if __name__ == "__main__":
    main()
