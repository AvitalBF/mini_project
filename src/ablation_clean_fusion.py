#!/usr/bin/env python
"""Stage 3 follow-up ablation: "Clean Anatomical Routing".

Purpose
-------
The first fusion ablation (src/ablation_fusion.py) concatenated raw SAMv2 feature vectors with
SPECTRE features. This follow-up removes every SAMv2 feature vector. SAMv2 is used only as the
anatomical coordinate router that already produced the six regional SPECTRE embeddings stored in
outputs/spectre_samv2_fusion.npz (`spectre_regional`, shape (N, 6, 1080)). We test whether these
SPECTRE-only regional embeddings add signal on top of the whole-scan SPECTRE embedding
(`spectre_global`, shape (N, 1080)). No features are re-extracted.

Evaluation protocol (identical to evaluate_embeddings.py and ablation_fusion.py)
-------------------------------------------------------------------------------
* Labels: `labels` from the NPZ, mapped by evaluate_embeddings.make_tasks.
* Tasks: `binary` (CT-0 vs CT-1..4) and `multiclass4` (CT-0, CT-1, CT-2, CT-3+4). We skip the
  5-class `multiclass` task on purpose (CT-4 has only 2 scans) to keep runtime short.
* Outer CV: 5-fold StratifiedKFold (shuffle=True) x 5 repeats, random_state = 42..46, which gives
  25 outer folds. The splits depend only on (y, seed), so every configuration and regime is scored
  on the same 25 test folds. This makes paired tests across configurations valid.
* Classifiers: LogisticRegression(class_weight="balanced", max_iter=5000) and
  LinearSVC(class_weight="balanced", max_iter=20000), with random_state = the base seed (42), as in
  evaluate_embeddings.make_models.
* Metrics: evaluate_embeddings.metrics (accuracy, macro-F1, one-vs-rest ROC-AUC) on each outer
  test fold, using evaluate_embeddings.scores_full for continuous scores.

No data leakage
---------------
Every data-dependent step is fitted on the outer training fold only: StandardScaler, PCA, the
choice of C (Regime B) and the late-fusion weight alpha. Steps are wrapped as sklearn
estimators, so the inner 3-fold searches also refit scalers and PCA on each inner training split.
The only transform done before CV is the per-scan mean-centering of the six regional embeddings
(see the residual configurations below). It uses only that scan's own six vectors and no
statistics from other scans, so it cannot leak information between folds.

Feature configurations (no SAMv2 feature vectors in any configuration)
---------------------------------------------------------------------
SPECTRE_global            1080  Whole-scan embedding. StandardScaler -> classifier. Baseline;
                                Regime A reproduces evaluate_embeddings.py exactly.
SPECTRE_regional_PCA64      64  Flattened regional embeddings (6 x 1080 = 6480) -> PCA(64)
                                -> StandardScaler -> classifier. Measures the regional signal on
                                its own.
Clean_Anat_PCA{32,64}  1080+k   [L2(Scale(global)) , L2(Scale(PCA_k(regional_flat)))].
                                Each block is standardized and then L2-normalized row-wise on its
                                own, so the 1080-d global block cannot swamp the k-d regional
                                block (and the reverse).
Clean_Residual_PCA{32,64} 1080+k  Same per-block L2 fusion, but the regional block is built from
                                *spatial residuals*: regional_i - mean_j(regional_j) for each scan
                                (mean-centered across the six lung zones). This removes the shared
                                scan-level component, so the block holds only the differences
                                between lung zones. We chose mean-centering over
                                (regional_i - global) because the regional embeddings are pooled
                                differently from the global one: on this dataset the mean
                                ||regional_i - global|| (~44) is larger than ||global|| (~33),
                                whereas the mean-centered residual norm is ~8.6. So subtracting
                                the global embedding would mostly measure the pooling mismatch,
                                not spatial heterogeneity.
Late_Fusion_Prob                Two classifiers: one on SPECTRE_global and one on
                                SPECTRE_regional_PCA64. Their scores are blended as
                                alpha * s_global + (1 - alpha) * s_regional. Scores are
                                predict_proba for LogReg. For LinearSVM they are decision_function
                                values divided by their training-set standard deviation. We divide
                                by the scale only and do not subtract the mean, so each model's
                                zero decision threshold and per-class argmax ordering stay intact.
                                alpha is chosen from [0.5, 0.6, ..., 1.0] by inner 3-fold
                                StratifiedKFold on the outer training fold (mean macro-F1; ties go
                                to the larger alpha, i.e. toward the baseline).

Regimes
-------
A (fixed baseline protocol): LogReg C=1.0, LinearSVM C=0.01, the values used by
  evaluate_embeddings.py.
B (inner-CV tuned C): C is chosen from [1e-4, 1e-3, 1e-2, 1e-1, 1, 10] by inner 3-fold
  StratifiedKFold (macro-F1) on each outer training fold. The same procedure is applied to every
  configuration, including SPECTRE_global. For Late_Fusion_Prob, each sub-classifier's C is
  tuned independently in this way, and alpha is then selected with those C values.

Significance: paired t-test and Wilcoxon signed-rank test on the 25 matched outer folds. Each
variant is compared with SPECTRE_global under the same regime and the same classifier.

Outputs: outputs/clean_fusion_results.json (full per-fold metrics, selected hyperparameters
and tests) and outputs/clean_fusion_summary.md (formatted report).
"""

from __future__ import annotations

import os

# One BLAS thread per worker; parallelism comes from joblib over independent outer folds.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, normalize

from ablation_fusion import paired_tests
from evaluate_embeddings import make_models, make_tasks, metrics, scores_full

FUSION_NPZ_DEFAULT = Path("/workspace/outputs/spectre_samv2_fusion.npz")
FOLDS, REPEATS, SEED = 5, 5, 42          # 5 x 5 = 25 outer folds, seeds 42..46
INNER_FOLDS = 3                           # inner StratifiedKFold for C / alpha selection
C_FIXED = {"LogReg": 1.0, "LinearSVM": 0.01}
C_GRID = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
ALPHA_GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
MODEL_NAMES = ["LogReg", "LinearSVM"]
TASKS = ["binary", "multiclass4"]         # 5-class `multiclass` deliberately excluded
REGIMES = ["A_fixed", "B_tunedC"]
BASELINE = "SPECTRE_global"
D_GLOBAL = 1080


def _silence_warnings() -> None:
    """Suppress expected, non-informative warnings (also inside joblib worker processes)."""
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", message=".*least populated class.*")


# ======================================================================== feature construction
class BlockFeatures(BaseEstimator, TransformerMixin):
    """Fold-safe feature builder for the clean SPECTRE configurations.

    The input X is a column concatenation [global (1080) | regional block (6480)]. Either block can
    be switched off. All statistics (StandardScaler means/variances, PCA bases) are learned in
    `fit`, i.e. only on whatever training rows the enclosing Pipeline receives.

    Parameters
    ----------
    use_global : include the standardized global block.
    use_regional : include the PCA-compressed, standardized regional block.
    pca_k : number of PCA components for the regional block.
    l2_blocks : if True, L2-normalize each block row-wise on its own after standardization.
    random_state : PCA random state (randomized SVD solver).
    """

    def __init__(self, use_global=True, use_regional=True, pca_k=64, l2_blocks=True, random_state=42):
        self.use_global = use_global
        self.use_regional = use_regional
        self.pca_k = pca_k
        self.l2_blocks = l2_blocks
        self.random_state = random_state

    def fit(self, X, y=None):
        if self.use_global:
            self.scaler_global_ = StandardScaler().fit(X[:, :D_GLOBAL])
        if self.use_regional:
            self.pca_ = PCA(n_components=self.pca_k, random_state=self.random_state).fit(X[:, D_GLOBAL:])
            self.scaler_regional_ = StandardScaler().fit(self.pca_.transform(X[:, D_GLOBAL:]))
        return self

    def transform(self, X):
        blocks = []
        if self.use_global:
            blocks.append(self.scaler_global_.transform(X[:, :D_GLOBAL]))
        if self.use_regional:
            blocks.append(self.scaler_regional_.transform(self.pca_.transform(X[:, D_GLOBAL:])))
        if self.l2_blocks:
            blocks = [normalize(b, norm="l2") for b in blocks]
        return np.concatenate(blocks, axis=1)


def base_classifier(model_name: str, C: float, seed: int):
    """The bare classifier step from evaluate_embeddings.make_models with the requested C, so
    class_weight / max_iter / random_state always match the baseline script."""
    kwargs = {"LogReg": (C, 1.0), "LinearSVM": (1.0, C)}[model_name]
    return make_models(kwargs[0], kwargs[1], seed)[model_name].steps[-1][1]


def make_pipeline_for(config: str, model_name: str, C: float, seed: int) -> Pipeline:
    """Unfitted single-classifier pipeline for a non-late-fusion configuration."""
    if config == "SPECTRE_global":
        # Exactly evaluate_embeddings.make_models: StandardScaler -> classifier.
        kwargs = {"LogReg": (C, 1.0), "LinearSVM": (1.0, C)}[model_name]
        return make_models(kwargs[0], kwargs[1], seed)[model_name]
    if config == "SPECTRE_regional_PCA64":
        feats = BlockFeatures(use_global=False, use_regional=True, pca_k=64, l2_blocks=False)
    elif config.startswith(("Clean_Anat_PCA", "Clean_Residual_PCA")):
        k = int(config.split("PCA")[1].split("_")[0])
        feats = BlockFeatures(use_global=True, use_regional=True, pca_k=k, l2_blocks=True)
    else:
        raise ValueError(config)
    return Pipeline([("feats", feats), ("clf", base_classifier(model_name, C, seed))])


def maybe_tune_C(pipe: Pipeline, tune: bool, seed: int):
    """Regime B: wrap the pipeline in an inner 3-fold GridSearchCV over C (macro-F1). The whole
    pipeline (scalers + PCA + classifier) is refit on every inner training split."""
    if not tune:
        return pipe
    return GridSearchCV(pipe, {f"{pipe.steps[-1][0]}__C": C_GRID}, scoring="f1_macro",
                        cv=StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=seed),
                        refit=True, n_jobs=1, error_score="raise")


# ======================================================================== late fusion
class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """Score-level fusion of a global classifier and a regional-PCA64 classifier.

    The input X is [global (1080) | regional_flat (6480)]. Each sub-model sees only its own block.
    alpha is selected by inner StratifiedKFold on the training data passed to `fit`. In Regime B
    each sub-model's C is first tuned by its own inner GridSearchCV on the same training data.
    """

    def __init__(self, model_name="LogReg", tune_C=False, seed=42):
        self.model_name = model_name
        self.tune_C = tune_C
        self.seed = seed

    # ---------------------------------------------------------------- helpers
    def _global_pipe(self, C):
        return make_pipeline_for("SPECTRE_global", self.model_name, C, self.seed)

    def _regional_pipe(self, C):
        return make_pipeline_for("SPECTRE_regional_PCA64", self.model_name, C, self.seed)

    @staticmethod
    def _split(X):
        # The regional pipe's BlockFeatures expects [global | regional]; global columns are ignored.
        return X[:, :D_GLOBAL], X

    def _raw_scores(self, model, X):
        """(n, n_classes_seen) scores: probabilities for LogReg, or the decision function for
        LinearSVM (a binary 1-d decision d is expanded to [-d, d])."""
        if self.model_name == "LogReg":
            return model.predict_proba(X)
        s = model.decision_function(X)
        return np.stack([-s, s], axis=1) if s.ndim == 1 else s

    def _fit_pair(self, Xg, Xr, y, C_g, C_r):
        """Fit both sub-models; for LinearSVM, record the train-set decision scale."""
        mg = self._global_pipe(C_g).fit(Xg, y)
        mr = self._regional_pipe(C_r).fit(Xr, y)
        if self.model_name == "LinearSVM":
            sd_g = float(np.std(self._raw_scores(mg, Xg))) or 1.0
            sd_r = float(np.std(self._raw_scores(mr, Xr))) or 1.0
        else:
            sd_g = sd_r = 1.0
        return mg, mr, sd_g, sd_r

    def _blend(self, mg, mr, sd_g, sd_r, Xg, Xr, alpha):
        return alpha * self._raw_scores(mg, Xg) / sd_g + (1.0 - alpha) * self._raw_scores(mr, Xr) / sd_r

    def _select_C(self, pipe, X, y):
        gs = maybe_tune_C(pipe, True, self.seed).fit(X, y)
        return float(next(iter(gs.best_params_.values())))

    # ---------------------------------------------------------------- sklearn API
    def fit(self, X, y):
        Xg, Xr = self._split(X)
        self.classes_ = np.unique(y)
        if self.tune_C:
            self.C_global_ = self._select_C(self._global_pipe(1.0), Xg, y)
            self.C_regional_ = self._select_C(self._regional_pipe(1.0), Xr, y)
        else:
            self.C_global_ = self.C_regional_ = C_FIXED[self.model_name]

        # Inner CV for alpha: mean macro-F1 across inner folds for each candidate alpha.
        inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=self.seed)
        f1_by_alpha = np.zeros(len(ALPHA_GRID))
        for itr, ite in inner.split(Xg, y):
            mg, mr, sd_g, sd_r = self._fit_pair(Xg[itr], Xr[itr], y[itr], self.C_global_, self.C_regional_)
            cls = mg.classes_
            for i, a in enumerate(ALPHA_GRID):
                pred = cls[np.argmax(self._blend(mg, mr, sd_g, sd_r, Xg[ite], Xr[ite], a), axis=1)]
                f1_by_alpha[i] += f1_score(y[ite], pred, average="macro", zero_division=0) / INNER_FOLDS
        best = np.flatnonzero(np.isclose(f1_by_alpha, f1_by_alpha.max()))
        self.alpha_ = float(ALPHA_GRID[best[-1]])  # ties -> largest alpha (closest to baseline)
        self.inner_f1_by_alpha_ = f1_by_alpha.tolist()

        self.model_g_, self.model_r_, self.sd_g_, self.sd_r_ = self._fit_pair(
            Xg, Xr, y, self.C_global_, self.C_regional_)
        return self

    def _scores(self, X):
        Xg, Xr = self._split(X)
        return self._blend(self.model_g_, self.model_r_, self.sd_g_, self.sd_r_, Xg, Xr, self.alpha_)

    def predict(self, X):
        return self.classes_[np.argmax(self._scores(X), axis=1)]

    def decision_function(self, X):
        """Blended scores, shape (n, n_classes_seen). Consumed by evaluate_embeddings.scores_full
        (for binary tasks it takes column 1, the positive-class score)."""
        return self._scores(X)


# ======================================================================== evaluation
CONFIGS = [
    "SPECTRE_global",
    "SPECTRE_regional_PCA64",
    "Clean_Anat_PCA32_L2",
    "Clean_Anat_PCA64_L2",
    "Clean_Residual_PCA32_L2",
    "Clean_Residual_PCA64_L2",
    "Late_Fusion_Prob",
]
CONFIG_DIMS = {
    "SPECTRE_global": "1080",
    "SPECTRE_regional_PCA64": "64",
    "Clean_Anat_PCA32_L2": "1080 + 32",
    "Clean_Anat_PCA64_L2": "1080 + 64",
    "Clean_Residual_PCA32_L2": "1080 + 32",
    "Clean_Residual_PCA64_L2": "1080 + 64",
    "Late_Fusion_Prob": "1080 / 64 (two models)",
}
CONFIG_INPUT = {  # which precomputed raw matrix each configuration consumes
    "SPECTRE_global": "global",
    "SPECTRE_regional_PCA64": "global_regional",
    "Clean_Anat_PCA32_L2": "global_regional",
    "Clean_Anat_PCA64_L2": "global_regional",
    "Clean_Residual_PCA32_L2": "global_residual",
    "Clean_Residual_PCA64_L2": "global_residual",
    "Late_Fusion_Prob": "global_regional",
}


def build_estimator(config: str, model_name: str, regime: str, seed: int):
    tune = regime == "B_tunedC"
    if config == "Late_Fusion_Prob":
        return LateFusionClassifier(model_name=model_name, tune_C=tune, seed=seed)
    return maybe_tune_C(make_pipeline_for(config, model_name, C_FIXED[model_name], seed), tune, seed)


def selected_hparams(model) -> dict:
    """Hyperparameters chosen inside the outer training fold (for the JSON audit trail)."""
    if isinstance(model, LateFusionClassifier):
        return {"C_global": model.C_global_, "C_regional": model.C_regional_, "alpha": model.alpha_}
    if isinstance(model, GridSearchCV):
        return {"C": float(next(iter(model.best_params_.values())))}
    return {}


def run_fold(X, y, n_classes, tr, te, config, model_name, regime, seed):
    """Fit one configuration on one outer training fold and score it on the matching test fold."""
    _silence_warnings()
    m = build_estimator(config, model_name, regime, seed).fit(X[tr], y[tr])
    pred, s = m.predict(X[te]), scores_full(m, X[te], n_classes)
    return dict(te=te, pred=pred, scores=s, metrics=metrics(y[te], pred, s, n_classes),
                hparams=selected_hparams(m))


def outer_splits(y):
    """The exact 25 outer splits of evaluate_embeddings.evaluate (seeds 42..46)."""
    out = []
    for r in range(REPEATS):
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED + r)
        out.extend((r, tr, te) for tr, te in skf.split(np.zeros(len(y)), y))
    return out


def aggregate(fold_results, y, n_classes):
    """Fold-level mean/std, pooled out-of-fold metrics per repeat, and the mean confusion matrix,
    following the same conventions as ablation_fusion.evaluate_detailed."""
    per_fold = {k: [float(fr["metrics"][k]) for _, fr in fold_results] for k in ("accuracy", "macro_f1", "roc_auc")}
    pooled_rows, cms = [], []
    for r in range(REPEATS):
        oof_pred = np.empty(len(y), dtype=int)
        oof_s = np.empty(len(y)) if n_classes == 2 else np.empty((len(y), n_classes))
        for rr, fr in fold_results:
            if rr == r:
                oof_pred[fr["te"]], oof_s[fr["te"]] = fr["pred"], fr["scores"]
        pooled_rows.append(metrics(y, oof_pred, oof_s, n_classes))
        cms.append(confusion_matrix(y, oof_pred, labels=np.arange(n_classes)))

    def ms(vals):
        return [float(np.nanmean(vals)), float(np.nanstd(vals))]

    return dict(
        fold={k: ms(v) for k, v in per_fold.items()},
        pooled={k: ms([row[k] for row in pooled_rows]) for k in per_fold},
        confusion=np.mean(cms, axis=0).tolist(),
        per_fold=per_fold,
        selected_hparams=[fr["hparams"] for _, fr in fold_results],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", type=Path, nargs="?", default=FUSION_NPZ_DEFAULT)
    ap.add_argument("--json-out", type=Path, default=Path("/workspace/outputs/clean_fusion_results.json"))
    ap.add_argument("--md-out", type=Path, default=Path("/workspace/outputs/clean_fusion_summary.md"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=False)
    labels = d["labels"].astype(int)
    n = len(labels)
    print(f"{args.npz}: {n} scans, class counts CT-0..4: {np.bincount(labels, minlength=5).tolist()}")

    g = d["spectre_global"].astype(np.float64)                             # (N, 1080)
    reg = d["spectre_regional"].astype(np.float64)                         # (N, 6, 1080)
    resid = reg - reg.mean(axis=1, keepdims=True)                          # per-scan zone centering
    inputs = {
        "global": g,
        "global_regional": np.concatenate([g, reg.reshape(n, -1)], axis=1),    # (N, 7560)
        "global_residual": np.concatenate([g, resid.reshape(n, -1)], axis=1),  # (N, 7560)
    }
    tasks = make_tasks(labels)

    # One flat job list over (task, config, model, regime, outer fold) for load balancing.
    jobs, keys = [], []
    for task in TASKS:
        y, class_names = tasks[task]
        n_classes = len(class_names)
        for r, tr, te in outer_splits(y):
            for config in CONFIGS:
                for model_name in MODEL_NAMES:
                    for regime in REGIMES:
                        keys.append((task, config, model_name, regime, r))
                        jobs.append(delayed(run_fold)(inputs[CONFIG_INPUT[config]], y, n_classes, tr, te,
                                                      config, model_name, regime, SEED))
    print(f"[eval] {len(jobs)} outer-fold fits "
          f"({len(TASKS)} tasks x {len(CONFIGS)} configs x {len(MODEL_NAMES)} models x "
          f"{len(REGIMES)} regimes x {FOLDS * REPEATS} folds)", flush=True)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        _silence_warnings()
        outs = Parallel(n_jobs=args.n_jobs, verbose=5)(jobs)
    runtime = time.perf_counter() - t0
    print(f"[eval] total {runtime:.1f}s", flush=True)

    # Group the fold results (job order preserves the fold order within each group).
    grouped: dict = {}
    for (task, config, model_name, regime, r), out in zip(keys, outs):
        grouped.setdefault((regime, config, task, model_name), []).append((r, out))

    results: dict = {}
    for (regime, config, task, model_name), frs in grouped.items():
        y, class_names = tasks[task]
        results.setdefault(regime, {}).setdefault(config, {}).setdefault(task, {})[model_name] = \
            aggregate(frs, y, len(class_names))

    # Paired tests vs the matched baseline (same regime, task, model, folds).
    significance: dict = {}
    for regime in REGIMES:
        for config in CONFIGS:
            if config == BASELINE:
                continue
            for task in TASKS:
                for model_name in MODEL_NAMES:
                    a = results[regime][config][task][model_name]["per_fold"]
                    b = results[regime][BASELINE][task][model_name]["per_fold"]
                    significance.setdefault(regime, {}).setdefault(config, {}).setdefault(task, {})[model_name] = {
                        metric: paired_tests(a[metric], b[metric]) for metric in ("macro_f1", "roc_auc", "accuracy")
                    }

    for regime in REGIMES:
        for config in CONFIGS:
            for task in TASKS:
                for model_name in MODEL_NAMES:
                    mf1 = results[regime][config][task][model_name]["fold"]["macro_f1"]
                    print(f"[{regime}] {config:<24} {task:<11} {model_name:<10} macro_f1={mf1[0]:.3f}+/-{mf1[1]:.3f}")

    payload = dict(
        protocol=dict(
            outer_folds=FOLDS, repeats=REPEATS, seeds=list(range(SEED, SEED + REPEATS)),
            total_folds=FOLDS * REPEATS, inner_folds=INNER_FOLDS, inner_scoring="f1_macro",
            C_fixed=C_FIXED, C_grid=C_GRID, alpha_grid=ALPHA_GRID, models=MODEL_NAMES, tasks=TASKS,
            skipped_tasks=["multiclass"], regimes=REGIMES, baseline=BASELINE, source_npz=str(args.npz),
            residual_definition="regional_i - mean_j(regional_j), per scan (mean-centered across 6 zones)",
            svm_score_standardization="decision_function / std(train decision_function), per sub-model",
            runtime_seconds=runtime,
        ),
        configs={c: {"dims": CONFIG_DIMS[c], "input": CONFIG_INPUT[c]} for c in CONFIGS},
        results=results,
        significance_vs_baseline=significance,
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2))
    print(f"[saved] {args.json_out}")

    write_markdown_report(args.md_out, results, significance, runtime)
    print(f"[saved] {args.md_out}")


# ======================================================================== report
def _fmt(ms) -> str:
    return f"{ms[0]:.3f} +/- {ms[1]:.3f}"


def _p(p) -> str:
    if p is None:
        return "n/a"
    stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
    return f"{p:.4f}{stars}"


def _hparam_summary(hp_list: list) -> str:
    """Compact description of the hyperparameters selected across the 25 outer folds."""
    if not hp_list or not hp_list[0]:
        return "fixed"
    parts = []
    for key in hp_list[0]:
        vals = [h[key] for h in hp_list]
        uniq, cnt = np.unique(vals, return_counts=True)
        mode = uniq[np.argmax(cnt)]
        label = {"C": "C", "C_global": "C_g", "C_regional": "C_r", "alpha": "alpha"}[key]
        parts.append(f"{label}: mode {mode:g} ({cnt.max()}/{len(vals)})")
    return "; ".join(parts)


def write_markdown_report(path: Path, results: dict, significance: dict, runtime: float) -> None:
    L = []
    L.append("# Stage 3 Follow-up: Clean Anatomical Routing (SPECTRE-only regional fusion)\n")
    L.append("## Protocol\n")
    L.append(f"- Source: `outputs/spectre_samv2_fusion.npz` (1110 scans; no re-extraction). "
             f"**No SAMv2 feature vectors are used.** SAMv2 served only as the anatomical "
             f"router that produced the six regional SPECTRE embeddings (`spectre_regional`, 6 x 1080).")
    L.append(f"- Outer CV: {FOLDS}-fold StratifiedKFold x {REPEATS} repeats (seeds {SEED}..{SEED + REPEATS - 1}, "
             f"{FOLDS * REPEATS} folds), identical splits to `evaluate_embeddings.py` / `ablation_fusion.py`.")
    L.append("- Tasks: `binary` (CT-0 vs CT-1..4) and `multiclass4` (CT-0 / CT-1 / CT-2 / CT-3+4). "
             "The 5-class `multiclass` task was intentionally skipped.")
    L.append("- Classifiers: LogReg (class_weight=balanced, max_iter=5000), LinearSVM "
             "(class_weight=balanced, max_iter=20000).")
    L.append("- **Regime A (fixed):** LogReg C=1.0, LinearSVM C=0.01 (original baseline protocol).")
    L.append(f"- **Regime B (tuned C):** C in {C_GRID} chosen by inner {INNER_FOLDS}-fold StratifiedKFold "
             f"(macro-F1) inside every outer training fold, for every configuration including the baseline.")
    L.append(f"- Late fusion: alpha in {ALPHA_GRID} chosen by inner {INNER_FOLDS}-fold StratifiedKFold "
             f"(macro-F1) inside every outer training fold, in both regimes; ties go to the larger alpha.")
    L.append("- Leakage control: StandardScaler, PCA, C and alpha are all fitted on the outer training fold "
             "only (and refit on every inner split). The only pre-CV transform, per-scan mean-centering "
             "of the six zone embeddings, uses no information from other scans.")
    L.append(f"- Wall-clock runtime: {runtime:.0f} s.\n")

    L.append("## Configurations\n")
    L.append("| Config | Dims | Description |")
    L.append("|---|---|---|")
    desc = {
        "SPECTRE_global": "Whole-scan SPECTRE embedding (baseline)",
        "SPECTRE_regional_PCA64": "Regional SPECTRE (6x1080) -> in-fold PCA(64); standalone regional signal",
        "Clean_Anat_PCA32_L2": "global + PCA32(regional); per-block StandardScaler + L2",
        "Clean_Anat_PCA64_L2": "global + PCA64(regional); per-block StandardScaler + L2",
        "Clean_Residual_PCA32_L2": "global + PCA32(zone residuals, mean-centered per scan); per-block Scaler + L2",
        "Clean_Residual_PCA64_L2": "global + PCA64(zone residuals, mean-centered per scan); per-block Scaler + L2",
        "Late_Fusion_Prob": "alpha * s(global model) + (1 - alpha) * s(regional PCA64 model); alpha via inner CV",
    }
    for c in CONFIGS:
        L.append(f"| {c} | {CONFIG_DIMS[c]} | {desc[c]} |")
    L.append("")

    regime_title = {"A_fixed": "Regime A: fixed C (LogReg C=1.0, LinearSVM C=0.01)",
                    "B_tunedC": "Regime B: inner-CV tuned C"}
    for regime in REGIMES:
        L.append(f"## {regime_title[regime]}\n")
        L.append("Fold-level mean +/- std over 25 folds. Delta = paired mean difference in macro-F1 vs "
                 "`SPECTRE_global` (same regime, task, model); p-values: paired t-test / Wilcoxon.\n")
        for task in TASKS:
            L.append(f"### Task: `{task}`\n")
            L.append("| Config | Model | Macro-F1 | ROC-AUC | Accuracy | dMacro-F1 | t p | Wilcoxon p | Selected hparams |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for model_name in MODEL_NAMES:
                for c in CONFIGS:
                    r = results[regime][c][task][model_name]
                    if c == BASELINE:
                        d_str, tp, wp = "(ref)", "", ""
                    else:
                        s = significance[regime][c][task][model_name]["macro_f1"]
                        d_str = f"{s['mean_diff']:+.4f}"
                        tp, wp = _p(s["t_p"]), _p(s["wilcoxon_p"])
                    L.append(f"| {c} | {model_name} | {_fmt(r['fold']['macro_f1'])} | {_fmt(r['fold']['roc_auc'])} | "
                             f"{_fmt(r['fold']['accuracy'])} | {d_str} | {tp} | {wp} | "
                             f"{_hparam_summary(r['selected_hparams'])} |")
            L.append("")

        L.append(f"### ROC-AUC and accuracy significance vs `{BASELINE}` ({regime})\n")
        L.append("| Config | Task | Model | dROC-AUC | t p | Wilcoxon p | dAccuracy | t p | Wilcoxon p |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for c in CONFIGS:
            if c == BASELINE:
                continue
            for task in TASKS:
                for model_name in MODEL_NAMES:
                    sa = significance[regime][c][task][model_name]["roc_auc"]
                    sc = significance[regime][c][task][model_name]["accuracy"]
                    L.append(f"| {c} | {task} | {model_name} | {sa['mean_diff']:+.4f} | {_p(sa['t_p'])} | "
                             f"{_p(sa['wilcoxon_p'])} | {sc['mean_diff']:+.4f} | {_p(sc['t_p'])} | {_p(sc['wilcoxon_p'])} |")
        L.append("")

    # Best variant per (regime, task, model) by mean macro-F1, for a quick read.
    L.append("## Best configuration per setting (by mean macro-F1)\n")
    L.append("| Regime | Task | Model | Best config | Macro-F1 | Baseline Macro-F1 | dMacro-F1 | t p | Wilcoxon p |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for regime in REGIMES:
        for task in TASKS:
            for model_name in MODEL_NAMES:
                best = max(CONFIGS, key=lambda c: results[regime][c][task][model_name]["fold"]["macro_f1"][0])
                bm = results[regime][best][task][model_name]["fold"]["macro_f1"][0]
                base = results[regime][BASELINE][task][model_name]["fold"]["macro_f1"][0]
                if best == BASELINE:
                    L.append(f"| {regime} | {task} | {model_name} | {best} | {bm:.3f} | {base:.3f} | +0.0000 | - | - |")
                else:
                    s = significance[regime][best][task][model_name]["macro_f1"]
                    L.append(f"| {regime} | {task} | {model_name} | {best} | {bm:.3f} | {base:.3f} | "
                             f"{s['mean_diff']:+.4f} | {_p(s['t_p'])} | {_p(s['wilcoxon_p'])} |")
    L.append("")
    L.append("Significance markers: p < 0.05 `*`, p < 0.01 `**`, p < 0.001 `***`. Differences are "
             "variant - SPECTRE_global (positive = variant better). No correction for multiple comparisons "
             "is applied; with 6 variants x 2 tasks x 2 models x 2 regimes = 48 comparisons per metric, "
             "isolated p-values near 0.05 should be read with caution.\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
