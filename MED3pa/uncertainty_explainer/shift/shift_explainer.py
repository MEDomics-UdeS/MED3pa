"""
shift_explainer.py
==================
Implementation of the **aggregate** (whole-profile, ``tolerance=0``)
covariate-shift and outcome-shift tests from

    Singh et al. "Who Experiences Large Model Decay and Why?"
    A Hierarchical Framework for Diagnosing Heterogeneous Performance Drift.
    ICML 2025.  https://arxiv.org/abs/2506.00756
    Code repo: https://github.com/jjfeng/shift

The implementation is self-contained: ``common.py::test_mmd`` /
``get_correlated_features``, used solely for the optional
``filter_independent_features`` covariate-test feature filter, are ported as
``_test_mmd`` / ``_get_correlated_features``. That filter defaults to an HSIC
independence test (``_test_hsic``); the original MMD test is kept as
``independence_test="mmd"``.
This implementation is written to be *numerically equivalent* to that repo's
``DetectorTestExplainer.test_aggregate()`` (in ``estimate_datashifter.py``),
not just a loose approximation of the paper's formulas: the detector-fitting
algorithms, the density-ratio model, the loss-function contract, and the
plug-in/one-step estimators are all ported line-for-line from that class so
that, given the same fitted source/target outcome models and the same
train/test split, both implementations produce the same aggregate estimates.

Mathematical / code correspondence to jjfeng/shift's DetectorTestExplainer
---------------------------------------------------------------------------
* ``_compute_Z``                ← ``estimate_datashifter.py::_compute_risk`` / ``common.py::compute_risk``
* ``_fit_density_model``        ← ``estimate_datashifter.py::_estimate_density_model`` (``common.py::get_density_model``)
* ``_density_ratio``            ← ``estimate_datashifter.py::_get_density_ratio_x`` (``common.py::get_density_ratio_from_classifier``)
* ``_fit_covariate_detectors``  ← ``estimate_datashifter.py::_estimate_detectors_covariate_aggregate`` /
                                   ``_estimate_detectors_covariate_fixed_omega`` (``detectors.py::DetectorCovariateShift``)
* ``_fit_outcome_detector``     ← ``estimate_datashifter.py::_estimate_detectors_cond_outcome_aggregate``
* ``_compute_correlated_features`` ← ``estimate_datashifter.py::_compute_correlated_features``
* ``_run_covariate_test``       ← ``estimate_datashifter.py::_test_covariate_aggregate``
* ``_run_outcome_test``         ← ``estimate_datashifter.py::_test_cond_outcome_aggregate``
* ``_bootstrap_pvalue``         ← ``common.py::get_multiplier_bootstrap_dist`` +
                                   ``decomp_explainer.py::ExplainerInference._run_bootstrap_test``
* ``AggregateShiftTester.__call__`` ← ``estimate_datashifter.py::test_aggregate()``

MED3pa integration
------------------
The module exposes two classes:

* ``AggregateShiftTester``: the numpy-only engine holding all of the above
  (fits nuisance models/detectors eagerly on train splits, evaluates on test
  splits). Usable standalone, e.g. for parity checks against jjfeng/shift.
* ``ShiftExplainer``: the ``ExplanationMethod`` used by
  ``MED3pa.uncertainty_explainer.explainer.Explainer``. The reference dataset
  is the *source* domain and the target dataset is the *target* domain; each
  call tests the patients of one profile (source patients selected with
  ``profile_path``, target patients given as ``x_q``, optionally with their
  labels ``y_q``). The error models are fitted once, globally, on the train
  parts of the whole datasets; the density-ratio model is either global or
  refitted per profile (``density_model_scope``); the detectors are fitted
  on the profile's train parts and the tests run on its test parts.
  ``f_model`` is the model being explained (its cross-entropy is the default
  loss). The outcome-shift test needs target labels, so it is skipped when
  ``y_q`` is not given.

Known, intentional simplification: only the *aggregate* (whole-profile) test
is implemented, not the detailed/subgroup decomposition (``DetectorTestExplainer``'s
``--combos`` feature-group search), and outcome models are always treated as
binary classifiers (matching ``DetectorTestExplainer``'s ``is_binary=True``
outcome models). As a single-pass alternative to a ``--combos`` search,
``explain_covariate_shift``/``explain_outcome_shift`` give per-feature
attribution for each shift type directly from the already-fitted nuisance
models (SHAP values of the density-ratio classifier, and the SHAP-value
*difference* between the source/target outcome models — "explanation
shift" — respectively), at the cost of a ranking rather than a formal
per-subset hypothesis test.

Dependencies: numpy, pandas, scikit-learn required; ``shap`` optional, only
needed for ``explain_covariate_shift``/``explain_outcome_shift``; ``torch`` and
``pytorch-ignite`` optional, only needed for ``filter_independent_features=True``
with ``independence_test="mmd"``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from MED3pa.datasets import MaskedDataset
from MED3pa.uncertainty_explainer.abstract_methods import CovariateConceptShiftDetector

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ShiftTestResult:
    """Holds results for one shift type (covariate or outcome).

    Attributes
    ----------
    shift_type : str
        ``"covariate"`` or ``"outcome"``.
    estimate_plugin : float
        Plug-in (biased) estimate of the Maximum conditional Expectation of
        Exceedence (McEE).
    estimate_onestep : float
        One-step / AIPW-corrected estimate. ``None`` when the density model
        is unavailable and the correction requires it.
    pvalue : float
        Gaussian multiplier bootstrap p-value for the one-step estimate
        (matches ``ExplainerInference``'s ``onestep`` row).
    detected : bool
        True when ``pvalue < significance_level`` and (for a one-sided test)
        ``estimate_onestep > tolerance``.
    n_test : int
        Number of test-set rows used.
    details : dict
        Extra diagnostics, including ``pvalue_plugin`` (the plug-in
        estimate's bootstrap p-value) and detector/density-ratio stats.
    """

    shift_type: str
    estimate_plugin: float
    estimate_onestep: Optional[float]
    pvalue: Optional[float]
    detected: bool
    n_test: int
    details: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"ShiftTestResult(type={self.shift_type!r}, "
            f"plugin={self.estimate_plugin:.4f}, "
            f"onestep={self.estimate_onestep}, "
            f"pvalue={self.pvalue}, "
            f"detected={self.detected})"
        )

    def to_dict(self) -> dict:
        return asdict(self)


def _cross_entropy_loss(y: np.ndarray, proba: np.ndarray) -> np.ndarray:
    """Binary cross-entropy loss per sample."""
    eps = 1e-12
    p = np.clip(proba, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ---------------------------------------------------------------------------
# Feature filter: MMD test (ported from jjfeng/shift's common.py) and HSIC test
# ---------------------------------------------------------------------------

def _test_mmd(X_tr: np.ndarray, X_te: np.ndarray, alpha: float = 0.05, n_permutations: int = 1000,
              rng: Optional[np.random.Generator] = None) -> Tuple[bool, float, float]:
    """MMD two-sample test with a fixed bandwidth chosen by the median
    heuristic, using ignite's ``MaximumMeanDiscrepancy``. Follows
    ``common.py::test_mmd``, which used ``torch_two_sample.MMDStatistic``:

    * the kernel ``exp(-||x-y||² / median_dist)`` (torch_two_sample's
      ``alphas=[1/median_dist]``) is reproduced with ignite's
      ``exp(-||x-y||² / 2σ²)`` by setting ``var = σ² = median_dist / 2``;
    * ignite only computes the statistic, so the p-value is obtained with a
      permutation test on the pooled sample, like ``MMDStatistic.pval``
      (1000 permutations by default).

    ignite requires both samples to have the same shape.
    """
    try:
        import torch
        from ignite.metrics import MaximumMeanDiscrepancy
    except ImportError as e:
        raise ImportError(
            "independence_test='mmd' requires `torch` and `pytorch-ignite` "
            "(`pip install pytorch-ignite==0.5.5`)."
        ) from e

    assert len(X_tr.shape) == len(X_te.shape), "X_tr and X_te should have the same dimensions"
    if len(X_tr.shape) == 1:
        X_tr = X_tr.reshape(-1, 1)
        X_te = X_te.reshape(-1, 1)
    if X_tr.shape != X_te.shape:
        raise ValueError(f"X_tr and X_te must have the same shape, got {X_tr.shape} != {X_te.shape}.")
    rng = np.random.default_rng() if rng is None else rng

    X_tr = X_tr.astype(np.float32)
    X_te = X_te.astype(np.float32)

    all_dist = distance.cdist(X_tr, X_te, "euclidean")
    median_dist = np.median(all_dist)
    mmd_metric = MaximumMeanDiscrepancy(var=median_dist / 2)

    def mmd(x: np.ndarray, y: np.ndarray) -> float:
        mmd_metric.reset()
        mmd_metric.update((torch.from_numpy(x), torch.from_numpy(y)))
        return mmd_metric.compute()

    t_val = mmd(X_tr, X_te)

    # permutation test on the pooled sample
    n = len(X_tr)
    pooled = np.concatenate([X_tr, X_te])
    permuted_stats = np.empty(n_permutations)
    for k in range(n_permutations):
        permuted = pooled[rng.permutation(len(pooled))]
        permuted_stats[k] = mmd(permuted[:n], permuted[n:])
    pvalue = float(np.mean(permuted_stats >= t_val))

    decision = pvalue < alpha
    return decision, pvalue, t_val


def _centered_rbf_gram(x: np.ndarray) -> np.ndarray:
    """Doubly-centered Gaussian RBF Gram matrix HKH of `x`, with bandwidth σ
    set to the median pairwise distance (median heuristic)."""
    dist = distance.cdist(x, x, "euclidean")
    pairwise = dist[np.triu_indices(len(x), k=1)]
    sigma = np.median(pairwise) if pairwise.size else 0.0
    if sigma == 0.0:
        # e.g. a mostly-constant discrete feature: fall back to the mean non-zero distance
        positive = pairwise[pairwise > 0]
        if positive.size == 0:
            return np.zeros_like(dist)  # constant variable: HSIC = 0
        sigma = positive.mean()
    K = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return K - K.mean(axis=0, keepdims=True) - K.mean(axis=1, keepdims=True) + K.mean()


def _test_hsic(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05, n_permutations: int = 1000,
               rng: Optional[np.random.Generator] = None) -> Tuple[bool, float, float]:
    """HSIC independence test (Gretton et al. 2008) between the paired
    samples `X` and `Y`, with Gaussian kernels (median heuristic bandwidths).

    The statistic is the biased estimator HSIC_b = tr(KHLH) / n², and the
    p-value is obtained by permuting `Y` (which breaks the pairing, i.e.
    simulates independence).
    """
    X = X.reshape(-1, 1) if X.ndim == 1 else X
    Y = Y.reshape(-1, 1) if Y.ndim == 1 else Y
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same number of samples, got {len(X)} != {len(Y)}.")
    rng = np.random.default_rng() if rng is None else rng

    n = len(X)
    Kc = _centered_rbf_gram(X.astype(float))
    Lc = _centered_rbf_gram(Y.astype(float))

    # tr(KHLH) = sum(HKH * HLH) since H is idempotent and the matrices are symmetric;
    # permuting Y permutes rows/columns of HLH (H is permutation-invariant)
    t_val = float(np.sum(Kc * Lc) / n ** 2)
    permuted_stats = np.empty(n_permutations)
    for k in range(n_permutations):
        perm = rng.permutation(n)
        permuted_stats[k] = np.sum(Kc * Lc[np.ix_(perm, perm)]) / n ** 2
    pvalue = float(np.mean(permuted_stats >= t_val))

    decision = pvalue < alpha
    return decision, pvalue, t_val


_INDEPENDENCE_TESTS = {"hsic": _test_hsic, "mmd": _test_mmd}


def _get_correlated_features(X: np.ndarray, Y: np.ndarray, alpha: float = 0.05, test: str = "hsic",
                             n_permutations: int = 1000,
                             random_state: Optional[int] = None) -> Tuple[np.ndarray, list]:
    """Features correlated with the outcome. Returns a boolean mask of the
    correlated features and the p-values of the tests.

    `test` is either ``"hsic"`` (default, a true independence test between
    each feature and `Y`) or ``"mmd"`` (``common.py::get_correlated_features``:
    an MMD two-sample test between the values of each feature and the values
    of `Y`, which compares their marginal distributions rather than testing
    their dependence).
    """
    if test not in _INDEPENDENCE_TESTS:
        raise ValueError(f"Unknown independence test {test!r}, expected one of {list(_INDEPENDENCE_TESTS)}.")
    test_func = _INDEPENDENCE_TESTS[test]
    rng = np.random.default_rng(random_state)
    correlated_features_mask = np.zeros(X.shape[1], dtype=bool)
    pvalues = []
    for i in np.arange(X.shape[1]):
        decision, pval, _ = test_func(X[:, i], Y, alpha=alpha, n_permutations=n_permutations, rng=rng)
        correlated_features_mask[i] = decision
        pvalues.append(pval)
    return correlated_features_mask, pvalues


# Defaults mirroring estimate_datashifter.py / common.py constants
_OUTCOME_MODEL_ARGS = dict(max_depth=16, max_features="sqrt", n_estimators=400, bootstrap=True)
_DENSITY_MODEL_ARGS = dict(max_depth=16, max_features=None, n_estimators=400)
_MAX_DETECTORS_COV = 5
_NUM_OMEGA = 2000
_NUM_LAMBDA = 500
_OMEGA_MAX = 100.0
_LAMBDA_RANGE = (-5.0, 5.0)


# ---------------------------------------------------------------------------
# Nuisance model recipes (shared by AggregateShiftTester and ShiftExplainer)
# ---------------------------------------------------------------------------

def _fit_outcome_model(X_tr: np.ndarray, y_tr: np.ndarray, is_oracle: bool = False):
    """Fit an outcome model with DetectorTestExplainer's default
    non-grid-search recipe. Matches ``common.py::get_outcome_model(
    is_binary=True, do_grid_search=False)``'s non-grid-search branch.
    """
    if is_oracle:
        clf = LogisticRegression(penalty=None)
    else:
        clf = RandomForestClassifier(
            criterion="log_loss", n_jobs=-1, random_state=0, **_OUTCOME_MODEL_ARGS
        )
    clf.fit(X_tr, (y_tr > 0.5).astype(int))
    return clf


def _make_density_template(density_classifier=None, random_state: Optional[int] = 42):
    """Unfitted source-vs-target classifier for the density-ratio model.
    Defaults to ``common.py::get_density_model()``'s non-grid-search branch."""
    if density_classifier is None:
        return RandomForestClassifier(n_jobs=-1, random_state=0, **_DENSITY_MODEL_ARGS)
    if isinstance(density_classifier, type):
        return density_classifier(random_state=random_state)
    return density_classifier


def _fit_density_model(template, X_src: np.ndarray, X_tgt: np.ndarray):
    """Calibrated source (D=0) vs target (D=1) classifier.
    Matches ``estimate_datashifter.py::_estimate_density_model``."""
    n_src, n_tgt = len(X_src), len(X_tgt)
    X_pool = np.vstack([X_src, X_tgt])
    d_pool = np.array([0] * n_src + [1] * n_tgt)
    clf = CalibratedClassifierCV(template, cv=3)
    clf.fit(X_pool, d_pool)
    return clf


def _hash_split(X: np.ndarray, train_fraction: float, seed: Optional[int]) -> np.ndarray:
    """Boolean train mask assigning each row to train/test from a hash of its
    values (and the seed). The assignment of a row does not depend on which
    other rows are present, so a profile's split is exactly the restriction of
    the global split to that profile."""
    X = np.ascontiguousarray(X, dtype=np.float64)
    salt = str(seed).encode()
    u = np.array([
        int.from_bytes(hashlib.blake2b(salt + row.tobytes(), digest_size=8).digest(), "little")
        for row in X
    ], dtype=np.float64) / 2.0 ** 64
    return u < train_fraction


# ---------------------------------------------------------------------------
# Engine (numpy only)
# ---------------------------------------------------------------------------

class AggregateShiftTester:
    """Aggregate covariate-shift and outcome-shift tester, numerically
    matching ``jjfeng/shift``'s ``DetectorTestExplainer.test_aggregate()``.

    Parameters
    ----------
    source_error_model, target_error_model : sklearn classifier, optional
        Already-fitted classifiers with ``.predict_proba(X)`` returning
        P(Y=1|X) for the source / target domain (μ_0, μ_1). These play
        exactly the role of ``DetectorTestExplainer.source_outcome_model`` /
        ``target_outcome_model`` — pass the *same fitted instances* used by
        ``DetectorTestExplainer`` for a directly comparable computation. If
        not supplied, they are fitted internally from ``source_train_data`` /
        ``target_train_data`` (matching ``get_outcome_model(is_binary=True,
        is_oracle=...)``). The target model is only used by
        ``explain_outcome_shift`` and is fitted lazily there.
    source_train_data, target_train_data : tuple (X, y) or None
        Training data for each domain. Required (both) to fit the
        density-ratio model and the aggregate detectors — i.e. required for
        anything beyond the plug-in outcome estimate. The target ``y`` may be
        ``None``: the covariate test only needs target covariates, while the
        outcome detector (and thus the outcome test) needs target labels.
    feature_names : list[str], optional
        Feature names, used to label the attribution tables.
    ml_mdl : sklearn classifier, optional
        The model being explained, f(X). Used only to build the default
        ``loss_func`` (binary cross-entropy of ``ml_mdl``'s own predictions)
        when ``loss_func`` is not supplied.
    loss_func : callable(X, y) -> ndarray, optional
        Per-sample loss ℓ(y, f(X)) of the model being explained, as a
        function of the covariates and the (possibly synthetic 0/1) label —
        the same contract as ``common.py::LossEvaluator.get_loss(x, y)`` /
        the ``loss_func`` argument to ``DetectorTestExplainer``. Pass the
        exact same callable used to construct ``DetectorTestExplainer`` for
        an exact match on any of its ``--loss`` choices. If omitted, and
        ``ml_mdl`` is not given either, defaults to cross-entropy of
        ``source_error_model``'s own predictions (self-referential; matches
        the common case where the model being explained IS the source
        outcome model).
    is_oracle : bool, default False
        When an error model must be fit internally, use
        ``LogisticRegression(penalty=None)`` (True) or the same
        ``RandomForestClassifier`` ``DetectorTestExplainer`` uses (False).
    density_classifier : sklearn estimator instance or class, optional
        Unfitted estimator for the source-vs-target density-ratio
        classifier, used when ``density_model`` is not given. Defaults to
        the same ``RandomForestClassifier`` used by
        ``common.py::get_density_model()`` (its non-grid-search branch).
    density_model, outcome_detector, covariate_detectors : optional
        Escape hatches to reuse already-fitted nuisance components instead
        of fitting this class's own (fixed-hyperparameter, no-grid-search)
        recipe — most useful for reproducing ``DetectorTestExplainer``
        exactly when it was run with ``--do-grid-search`` (which changes
        its density/outcome model architecture via ``GridSearchCV`` in ways
        this class does not replicate). Pass
        ``density_model=detector_test_explainer.density_x_model``,
        ``outcome_detector=detector_test_explainer.agg_detectors_y[0]``, and
        ``covariate_detectors=[(omega, d.min_lambda) for d, omega in
        detector_test_explainer.agg_detectors_x]`` (after calling
        ``detector_test_explainer.test_aggregate()``) for an exact match.
        Each is used as-is (never refit) when given; ``covariate_detectors``
        is a list of ``(omega, lambda)`` pairs.
    outcome_detector_cls, outcome_detector_kwargs :
        Regressor class/kwargs used to fit the aggregate outcome-shift
        detector on ``target_train_data`` when ``outcome_detector`` is not
        given. Defaults to ``GradientBoostingRegressor(max_depth=2,
        n_estimators=200)``, matching ``estimate_datashifter.py::DETECTORS``.
    filter_independent_features : bool, default False
        Matches ``--filter-independent-features``: before running the
        **covariate**-shift test only (the outcome test always uses every
        feature), drop features found independent of the realized
        source-train loss (the ``independence_test`` run per feature at
        significance level ``filter_significance_level``). When at least one feature is
        dropped, a separate source outcome model and density-ratio model are
        fit on just the kept features for the covariate test (matching
        ``DetectorTestExplainer._prepare_nuisance_models``); the outcome
        test is unaffected and keeps using the full-feature
        ``source_error_model``/``density_model``.
    filter_significance_level : float, optional
        Significance level for the feature-filtering independence test.
        Defaults to ``significance_level``.
    independence_test : {"hsic", "mmd"}, default "hsic"
        Test used by the feature filter. ``"hsic"`` is an HSIC permutation
        test of independence between each feature and the loss. ``"mmd"`` is
        ``common.py::get_correlated_features``'s MMD two-sample test between
        the values of each feature and the values of the loss (it compares
        their marginal distributions, so it keeps nearly every feature).
    n_permutations : int, default 1000
        Number of permutations for the feature-filter test's p-value.
    correlated_features : ndarray of bool, optional
        Escape hatch: an already-computed boolean feature mask (e.g.
        ``detector_test_explainer.correlated_features``), used as-is instead
        of running the independence test. Overrides
        ``filter_independent_features``.
    source_error_model_filter, density_model_filter : optional
        Escape hatches for the covariate test's filtered nuisance models
        (e.g. ``detector_test_explainer.source_outcome_filter_model`` /
        ``.density_x_filter_model``), used as-is instead of being fit
        internally. When ``correlated_features`` keeps every feature, these
        default to ``source_error_model``/``density_model`` (no separate fit
        needed), matching ``DetectorTestExplainer``.
    tolerance : float, default 0.0
        τ — minimum shift magnitude to detect.
    prevalence : float, default 0.0
        Minimum detected-subgroup prevalence (in both domains, on train
        data) required to keep a candidate covariate-shift detector.
    significance_level : float, default 0.05
    n_bootstrap : int, default 10000
        Number of Gaussian multiplier bootstrap draws.
    one_sided_test : bool, default True
        Matches ``--one-sided-test``: whether the bootstrap p-value tests
        H0: McEE ≤ τ (one-sided) or |McEE| = 0 (two-sided).
    test_greater : bool, default True
        Matches ``--do-test-greater``: sign of the residual
        (``residual_sign = 1 if test_greater else -1``).
    num_omega, num_lambda, omega_max, lambda_range, max_detectors_cov :
        Grid-search parameters for the covariate-shift detector, matching
        ``DetectorTestExplainer.candidate_omegas`` / ``candidate_lambdas`` /
        ``MAX_DETECTORS_COV`` (defaults are exactly those values).
    do_clipping : bool, default False
        Matches ``--do-clipping``: clip predicted domain probabilities to
        [1e-3, 1-1e-3] instead of [1e-10, 1-1e-10] when computing the
        density ratio.
    random_state : int or None, default 42
        Used for the internally-fit density classifier (when
        ``density_classifier`` is a class) and for the bootstrap RNG.

    Notes
    -----
    *Eager fitting*: the density-ratio model and both aggregate detectors are
    fitted at construction time (from ``source_train_data`` /
    ``target_train_data``), so ``__call__`` is pure inference on whatever
    test split is passed in — mirroring how ``DetectorTestExplainer``
    fits nuisance models/detectors once on its train split and evaluates
    them on its held-out test split.
    """

    def __init__(
        self,
        *,
        source_error_model=None,
        target_error_model=None,
        source_train_data: Optional[Tuple] = None,
        target_train_data: Optional[Tuple] = None,
        feature_names: Optional[list] = None,
        ml_mdl=None,
        loss_func: Optional[Callable] = None,
        is_oracle: bool = False,
        density_classifier=None,
        density_model=None,
        outcome_detector=None,
        covariate_detectors: Optional[List[Tuple[float, float]]] = None,
        outcome_detector_cls=None,
        outcome_detector_kwargs: Optional[dict] = None,
        filter_independent_features: bool = False,
        filter_significance_level: Optional[float] = None,
        independence_test: str = "hsic",
        n_permutations: int = 1000,
        correlated_features: Optional[np.ndarray] = None,
        source_error_model_filter=None,
        density_model_filter=None,
        tolerance: float = 0.0,
        prevalence: float = 0.0,
        significance_level: float = 0.05,
        n_bootstrap: int = 10000,
        one_sided_test: bool = True,
        test_greater: bool = True,
        num_omega: int = _NUM_OMEGA,
        num_lambda: int = _NUM_LAMBDA,
        omega_max: float = _OMEGA_MAX,
        lambda_range: Tuple[float, float] = _LAMBDA_RANGE,
        max_detectors_cov: int = _MAX_DETECTORS_COV,
        do_clipping: bool = False,
        random_state: Optional[int] = 42,
        clip_density: Optional[float] = None,
    ):
        self.feature_names = feature_names
        self.is_oracle = is_oracle
        self.tolerance = tolerance
        self.prevalence = prevalence
        self.significance_level = significance_level
        self.n_bootstrap = n_bootstrap
        self.one_sided_test = one_sided_test
        self.test_greater = test_greater
        self.num_omega = num_omega
        self.num_lambda = num_lambda
        self.omega_max = omega_max
        self.lambda_range = lambda_range
        self.max_detectors_cov = max_detectors_cov
        self.do_clipping = do_clipping
        self.random_state = random_state
        self._density_eps = clip_density if clip_density is not None else (1e-3 if do_clipping else 1e-10)
        self._omega_grid = np.linspace(0.0, omega_max, num_omega)

        # ---- density-ratio classifier template ----
        self._density_clf_template = _make_density_template(density_classifier, random_state)

        # ---- outcome-shift detector template ----
        if outcome_detector_cls is None:
            outcome_detector_cls = GradientBoostingRegressor
        if outcome_detector_kwargs is None:
            outcome_detector_kwargs = dict(max_depth=2, n_estimators=200)
        self._outcome_detector_template = outcome_detector_cls(**outcome_detector_kwargs)

        # ---- unpack training data ----
        self._src_train = self._parse_data(source_train_data, "source_train_data")
        self._tgt_train = self._parse_data(target_train_data, "target_train_data")

        # ---- error models (never refit if given; the target one is only
        # needed by explain_outcome_shift, so it is resolved lazily there) ----
        self.source_error_model = self._resolve_error_model(source_error_model, self._src_train, "source")
        self.target_error_model = target_error_model

        # ---- loss function ℓ(y, f(X)), signature loss_func(X, y) ----
        self.ml_mdl = ml_mdl
        if loss_func is not None:
            self._loss_func = loss_func
        elif ml_mdl is not None:
            self._loss_func = lambda X, y: _cross_entropy_loss(y, ml_mdl.predict_proba(X)[:, 1])
        else:
            # Self-referential default: model being explained == source_error_model
            self._loss_func = lambda X, y: _cross_entropy_loss(y, self.source_error_model.predict_proba(X)[:, 1])

        # ---- domain ratio + density ratio model (Section C.1) ----
        # domain_ratio_train (the odds-scaling factor) always comes from the
        # train split sizes, whether or not density_model is injected.
        if self._src_train is not None and self._tgt_train is not None:
            self.domain_ratio_train = self._src_train[0].shape[0] / self._tgt_train[0].shape[0]
        else:
            self.domain_ratio_train = None

        if density_model is not None:
            # Never refit: reuse an externally-fitted density-ratio model
            # (e.g. DetectorTestExplainer.density_x_model) directly, so π(x)
            # matches exactly regardless of how it was fit (do_grid_search, ...).
            self.density_model = density_model
        elif self._src_train is not None and self._tgt_train is not None:
            self.density_model = self._fit_density_model(self._src_train[0], self._tgt_train[0])
        else:
            self.density_model = None

        # ---- optional feature filtering for the covariate test only
        # (estimate_datashifter.py::_prepare_nuisance_models /
        # _compute_correlated_features) ----
        self.filter_independent_features = filter_independent_features
        self.filter_significance_level = (
            filter_significance_level if filter_significance_level is not None else significance_level
        )
        if independence_test not in _INDEPENDENCE_TESTS:
            raise ValueError(
                f"Unknown independence_test {independence_test!r}, expected one of {list(_INDEPENDENCE_TESTS)}."
            )
        self.independence_test = independence_test
        self.n_permutations = n_permutations
        if correlated_features is not None:
            self.correlated_features = np.asarray(correlated_features, dtype=bool)
            self.correlated_feature_pvalues = None
        elif filter_independent_features:
            if self._src_train is None:
                raise ValueError(
                    "filter_independent_features=True requires source_train_data "
                    "to compute which features are correlated with the loss."
                )
            self.correlated_features, self.correlated_feature_pvalues = self._compute_correlated_features()
        elif self._src_train is not None:
            self.correlated_features = np.ones(self._src_train[0].shape[1], dtype=bool)
            self.correlated_feature_pvalues = None
        else:
            self.correlated_features = None
            self.correlated_feature_pvalues = None

        # Filtered source outcome model + density model for the covariate
        # test only (refit just when at least one feature is dropped,
        # matching DetectorTestExplainer; the target outcome model is never
        # filtered since the aggregate covariate test doesn't use it).
        all_features_kept = self.correlated_features is None or bool(np.all(self.correlated_features))
        if source_error_model_filter is not None:
            self.source_error_model_filter = source_error_model_filter
        elif all_features_kept:
            self.source_error_model_filter = self.source_error_model
        else:
            X_src_tr, y_src_tr = self._src_train
            self.source_error_model_filter = self._fit_default_error_model(
                X_src_tr[:, self.correlated_features], y_src_tr
            )

        if density_model_filter is not None:
            self.density_model_filter = density_model_filter
        elif all_features_kept:
            self.density_model_filter = self.density_model
        elif self._src_train is not None and self._tgt_train is not None:
            X_src_tr, _ = self._src_train
            X_tgt_tr, _ = self._tgt_train
            self.density_model_filter = self._fit_density_model(
                X_src_tr[:, self.correlated_features], X_tgt_tr[:, self.correlated_features]
            )
        else:
            self.density_model_filter = None

        # ---- aggregate detectors, fit eagerly on train data (or reuse
        # externally-fitted ones, e.g. from DetectorTestExplainer, for exact
        # parity independent of this class's own grid-search recipe) ----
        if covariate_detectors is not None:
            self._covariate_detectors = list(covariate_detectors)
        elif self.density_model_filter is not None and self._src_train is not None and self._tgt_train is not None:
            self._covariate_detectors = self._fit_covariate_detectors()
        else:
            self._covariate_detectors = None

        if outcome_detector is not None:
            self._outcome_detector = outcome_detector
        elif self._tgt_train is not None and self._tgt_train[1] is not None:
            self._outcome_detector = self._fit_outcome_detector()
        else:
            self._outcome_detector = None

    @property
    def residual_sign(self) -> int:
        return 1 if self.test_greater else -1

    # ------------------------------------------------------------------
    # Private helpers: construction
    # ------------------------------------------------------------------

    def _parse_data(self, data, name: str):
        if data is None:
            return None
        if not (isinstance(data, (tuple, list)) and len(data) == 2):
            raise ValueError(f"`{name}` must be a (X, y) tuple or None, got {type(data)}")
        X, y = data
        return np.asarray(X, dtype=float), (np.asarray(y, dtype=float) if y is not None else None)

    def _resolve_error_model(self, model, train_data, domain: str):
        """Return a fitted error model, fitting one if necessary."""
        if model is not None:
            return model
        if train_data is None or train_data[1] is None:
            raise ValueError(
                f"No `{domain}_error_model` was given and no `{domain}_train_data` "
                f"was supplied to fit one. Provide at least one of them."
            )
        X_tr, y_tr = train_data
        return self._fit_default_error_model(X_tr, y_tr)

    def _fit_default_error_model(self, X_tr: np.ndarray, y_tr: np.ndarray):
        """Fit an outcome model with ``_fit_outcome_model``'s recipe. Also
        used to fit the covariate test's filtered source outcome model."""
        return _fit_outcome_model(X_tr, y_tr, is_oracle=self.is_oracle)

    def _compute_correlated_features(self) -> Tuple[np.ndarray, list]:
        """Features correlated with the realized source-train loss, for the
        covariate test's optional feature filter. Follows
        ``estimate_datashifter.py::_compute_correlated_features`` (one
        ``independence_test`` per feature via ``_get_correlated_features``).
        """
        X_src_tr, y_src_tr = self._src_train
        loss_src_tr = self._loss_func(X_src_tr, y_src_tr)
        return _get_correlated_features(X=X_src_tr, Y=loss_src_tr, alpha=self.filter_significance_level,
                                        test=self.independence_test, n_permutations=self.n_permutations,
                                        random_state=self.random_state)

    # ------------------------------------------------------------------
    # Density ratio model  (estimate_datashifter.py::_estimate_density_model)
    # ------------------------------------------------------------------

    def _fit_density_model(self, X_src: np.ndarray, X_tgt: np.ndarray):
        return _fit_density_model(self._density_clf_template, X_src, X_tgt)

    def _density_ratio(self, X: np.ndarray, model=None) -> np.ndarray:
        """π(x) = p(D=1|x)/p(D=0|x) · (n_source_train / n_target_train).

        The domain_ratio_train scaling matches
        ``common.py::get_density_ratio_from_classifier(scale=domain_ratio_train)``
        as called from ``estimate_datashifter.py::_get_density_ratio_x``.
        `model` defaults to ``self.density_model``; pass
        ``self.density_model_filter`` for the (feature-filtered) covariate test.
        """
        model = self.density_model if model is None else model
        if model is None:
            raise RuntimeError(
                "Density ratio model is unavailable because source_train_data "
                "and/or target_train_data were not supplied at construction. "
                "The covariate-shift test and the one-step correction for the "
                "outcome-shift test both require it."
            )
        proba = model.predict_proba(X)
        p1 = np.clip(proba[:, 1], self._density_eps, 1 - self._density_eps)
        p0 = 1.0 - p1
        return (p1 / p0) * self.domain_ratio_train

    # ------------------------------------------------------------------
    # Nuisance: conditional loss Z_d(x) = E_d[ℓ(Y,f(X))|X=x]
    # ------------------------------------------------------------------

    def _compute_Z(self, X_outcome: np.ndarray, X_mdl: np.ndarray, error_model) -> np.ndarray:
        """Z_d(x) = ℓ(0,f(x))·(1-μ_d(x)) + ℓ(1,f(x))·μ_d(x).

        Matches ``common.py::compute_risk`` exactly. `X_outcome` feeds the
        outcome model μ_d (a feature subset when the covariate test's
        feature filter is active); `X_mdl` feeds `loss_func` ℓ and is always
        the full feature set — mirroring ``DataLoader``'s X vs mdl_X split
        in ``estimate_datashifter.py`` (``subset_X`` "keeps mdl_X features
        constant"). Pass the same array for both when no filtering applies.
        """
        mu = error_model.predict_proba(X_outcome)[:, 1]
        loss1 = self._loss_func(X_mdl, np.ones(len(X_mdl)))
        loss0 = self._loss_func(X_mdl, np.zeros(len(X_mdl)))
        return loss1 * mu + loss0 * (1 - mu)

    # ------------------------------------------------------------------
    # Detector fitting (on TRAIN data, eagerly at construction)
    # ------------------------------------------------------------------

    def _fit_covariate_detectors(self) -> List[Tuple[float, float]]:
        """Grid-search (ω, λ) detectors on the train split.

        For each ω, λ is chosen (on source-train) to minimize the mean of
        the clipped-at-0 detected residual (``detectors.py::DetectorCovariateShift.fit``).
        Candidate (ω, λ) pairs are kept when both source- and target-train
        detected prevalence clear ``self.prevalence``, then the top
        ``max_detectors_cov`` by detected residual are kept
        (``_estimate_detectors_covariate_aggregate``). Returns a list of
        ``(omega, lambda)`` pairs, evaluated later on the test split.
        """
        mask = self.correlated_features
        X_src_tr_full, _ = self._src_train
        X_tgt_tr_full, _ = self._tgt_train
        X_src_tr = X_src_tr_full[:, mask] if mask is not None else X_src_tr_full
        X_tgt_tr = X_tgt_tr_full[:, mask] if mask is not None else X_tgt_tr_full

        Z0_src_tr = self._compute_Z(X_src_tr, X_src_tr_full, self.source_error_model_filter)
        Z0_tgt_tr = self._compute_Z(X_tgt_tr, X_tgt_tr_full, self.source_error_model_filter)
        pi_src_tr = self._density_ratio(X_src_tr, model=self.density_model_filter)
        pi_tgt_tr = self._density_ratio(X_tgt_tr, model=self.density_model_filter)

        lambdas = np.linspace(self.lambda_range[0], self.lambda_range[1], self.num_lambda)[:, np.newaxis]

        candidates = []
        with np.errstate(divide="ignore", invalid="ignore"):
            for omega in self._omega_grid:
                # shape (num_lambda, n_src_train)
                scores_src = (self.residual_sign * Z0_src_tr[np.newaxis, :] - lambdas) * (
                    pi_src_tr[np.newaxis, :] * omega - 1.0
                ) - self.tolerance
                mean_clipped = np.clip(scores_src, 0, None).mean(axis=1)
                lam_idx = int(np.argmin(mean_clipped))
                lam = float(lambdas[lam_idx, 0])

                detected_src = scores_src[lam_idx]
                prevalence_source = float((detected_src > 0).mean())
                # NOTE: intentionally not guarded against prevalence_source==0
                # (yields nan, matching DetectorTestExplainer's literal
                # `mean(detected_residual)/weighted_prevalence_source`, which
                # has no such guard either when --prevalence=0).
                norm_detected_residual = float(np.clip(detected_src, 0, None).mean() / prevalence_source)

                detected_tgt = (self.residual_sign * Z0_tgt_tr - lam) * (pi_tgt_tr * omega - 1.0) - self.tolerance
                prevalence_target = float((detected_tgt > 0).mean())

                if prevalence_source >= self.prevalence and prevalence_target >= self.prevalence:
                    candidates.append((norm_detected_residual, omega, lam))

        candidates.sort(key=lambda c: -c[0])
        return [(float(omega), float(lam)) for _, omega, lam in candidates[: self.max_detectors_cov]]

    def _fit_outcome_detector(self):
        """Fit a regressor to predict the realized outcome-shift residual on
        target-train, matching
        ``estimate_datashifter.py::_estimate_detectors_cond_outcome_aggregate``.
        """
        X_tgt_tr, y_tgt_tr = self._tgt_train
        target_loss_tr = self._loss_func(X_tgt_tr, y_tgt_tr)
        Z0_tgt_tr = self._compute_Z(X_tgt_tr, X_tgt_tr, self.source_error_model)
        residual = self.residual_sign * (target_loss_tr - Z0_tgt_tr) - self.tolerance
        detector = clone(self._outcome_detector_template)
        detector.fit(X_tgt_tr, residual)
        return detector

    # ------------------------------------------------------------------
    # Pooled masked mean / Gaussian multiplier bootstrap
    # (common.py::get_multiplier_bootstrap_dist + decomp_explainer.py::
    #  ExplainerInference._run_bootstrap_test, generalized to a list of
    #  candidate detectors whose influence/mask arrays get max-pooled)
    # ------------------------------------------------------------------

    @staticmethod
    def _max_masked_mean(ics: List[np.ndarray], masks: List[np.ndarray]) -> float:
        estims = [np.mean(ic[mask]) for ic, mask in zip(ics, masks) if mask.sum() > 0]
        return float(np.max(estims)) if estims else 0.0

    def _bootstrap_pvalue(
        self, ics: List[np.ndarray], masks: List[np.ndarray], estim: float, rng: np.random.Generator
    ) -> float:
        if not ics:
            return 1.0
        n = len(ics[0])
        multipliers = rng.standard_normal((n, self.n_bootstrap))
        draws = []
        for ic, mask in zip(ics, masks):
            if mask.sum() == 0:
                continue
            centered = ic[mask] - ic[mask].mean()
            draws.append((multipliers[mask].T @ centered) / mask.sum())
        if not draws:
            return 1.0
        bootstrap_dist = np.max(np.vstack(draws), axis=0)
        if self.one_sided_test:
            return float(np.mean(bootstrap_dist >= estim))
        return float(np.mean(np.abs(bootstrap_dist) >= np.abs(estim)))

    def _is_detected(self, estim: float, pvalue: Optional[float]) -> bool:
        if pvalue is None:
            return False
        if self.one_sided_test:
            return bool(pvalue < self.significance_level and estim > self.tolerance)
        return bool(pvalue < self.significance_level)

    # ------------------------------------------------------------------
    # Aggregate tests (estimate_datashifter.py::_test_cond_outcome_aggregate
    # / _test_covariate_aggregate), evaluated on the test split
    # ------------------------------------------------------------------

    def _run_outcome_test(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        Z0_src: np.ndarray,
        Z0_tgt: np.ndarray,
        source_loss: np.ndarray,
        target_loss: np.ndarray,
        pi_src: Optional[np.ndarray],
        rng: np.random.Generator,
    ) -> ShiftTestResult:
        n_src, n_tgt = len(X_src), len(X_tgt)

        if self._outcome_detector is None:
            raise RuntimeError(
                "The outcome-shift test requires target_train_data to fit the "
                "aggregate outcome detector. Provide it at construction time."
            )

        detection_target = self._outcome_detector.predict(X_tgt) > 0
        detected_residual_target = self.residual_sign * (target_loss - Z0_tgt) - self.tolerance

        plugin_ic = np.concatenate([np.zeros(n_src), detected_residual_target])
        plugin_mask = np.concatenate([np.zeros(n_src, dtype=bool), detection_target])
        plugin_estim = self._max_masked_mean([plugin_ic], [plugin_mask])
        pval_plugin = self._bootstrap_pvalue([plugin_ic], [plugin_mask], plugin_estim, rng)

        if pi_src is not None:
            detection_source = self._outcome_detector.predict(X_src) > 0
            with np.errstate(divide="ignore", invalid="ignore"):
                correction_source = -self.residual_sign * (source_loss - Z0_src) * pi_src
            onestep_ic = np.concatenate([correction_source, detected_residual_target])
            onestep_mask = np.concatenate([detection_source, detection_target])
            onestep_estim = self._max_masked_mean([onestep_ic], [onestep_mask])
            pval_onestep = self._bootstrap_pvalue([onestep_ic], [onestep_mask], onestep_estim, rng)
        else:
            onestep_estim = None
            pval_onestep = None

        return ShiftTestResult(
            shift_type="outcome",
            estimate_plugin=plugin_estim,
            estimate_onestep=onestep_estim,
            pvalue=pval_onestep,
            detected=self._is_detected(onestep_estim, pval_onestep) if onestep_estim is not None else False,
            n_test=n_src + n_tgt,
            details={
                "h_detected_fraction_target": float(detection_target.mean()),
                "Z0_mean_tgt": float(Z0_tgt.mean()),
                "pvalue_plugin": pval_plugin,
            },
        )

    def _run_covariate_test(
        self,
        X_src: np.ndarray,
        X_tgt: np.ndarray,
        source_loss: np.ndarray,
        rng: np.random.Generator,
    ) -> ShiftTestResult:
        n_src, n_tgt = len(X_src), len(X_tgt)

        if self._covariate_detectors is None:
            raise RuntimeError(
                "The covariate-shift test requires source_train_data AND "
                "target_train_data to fit the density-ratio model and the "
                "aggregate covariate detectors."
            )

        # Covariate test uses its own (optionally feature-filtered) outcome
        # model and density model — see filter_independent_features.
        mask = self.correlated_features
        X_src_f = X_src[:, mask] if mask is not None else X_src
        X_tgt_f = X_tgt[:, mask] if mask is not None else X_tgt
        Z0_src = self._compute_Z(X_src_f, X_src, self.source_error_model_filter)
        Z0_tgt = self._compute_Z(X_tgt_f, X_tgt, self.source_error_model_filter)
        pi_src = self._density_ratio(X_src_f, model=self.density_model_filter)
        pi_tgt = self._density_ratio(X_tgt_f, model=self.density_model_filter)

        plugin_ics, plugin_masks = [], []
        onestep_ics, onestep_masks = [], []
        n_detectors_used = 0

        with np.errstate(divide="ignore", invalid="ignore"):
            for omega, lam in self._covariate_detectors:
                score_src = (self.residual_sign * Z0_src - lam) * (pi_src * omega - 1.0) - self.tolerance
                score_tgt = (self.residual_sign * Z0_tgt - lam) * (pi_tgt * omega - 1.0) - self.tolerance
                source_mask = score_src > 0
                target_mask = score_tgt > 0
                n_detect_src, n_detect_tgt = int(source_mask.sum()), int(target_mask.sum())
                if n_detect_src + n_detect_tgt == 0:
                    continue
                n_detectors_used += 1
                dp_tgt = n_detect_tgt / (n_detect_src + n_detect_tgt)
                dp_src = 1.0 - dp_tgt

                plugin_ic = np.concatenate([
                    (-self.residual_sign * source_loss - self.tolerance) / dp_src,
                    (self.residual_sign * Z0_tgt) / dp_tgt,
                ])
                onestep_ic = np.concatenate([
                    (-self.residual_sign * (source_loss - (source_loss - Z0_src) * pi_src) - self.tolerance) / dp_src,
                    (self.residual_sign * Z0_tgt) / dp_tgt,
                ])
                detect_mask = np.concatenate([source_mask, target_mask])

                plugin_ics.append(plugin_ic)
                plugin_masks.append(detect_mask)
                onestep_ics.append(onestep_ic)
                onestep_masks.append(detect_mask)

        plugin_estim = self._max_masked_mean(plugin_ics, plugin_masks)
        onestep_estim = self._max_masked_mean(onestep_ics, onestep_masks)
        pval_plugin = self._bootstrap_pvalue(plugin_ics, plugin_masks, plugin_estim, rng)
        pval_onestep = self._bootstrap_pvalue(onestep_ics, onestep_masks, onestep_estim, rng)

        return ShiftTestResult(
            shift_type="covariate",
            estimate_plugin=plugin_estim,
            estimate_onestep=onestep_estim,
            pvalue=pval_onestep,
            detected=self._is_detected(onestep_estim, pval_onestep),
            n_test=n_src + n_tgt,
            details={
                "n_detectors": n_detectors_used,
                "pi_src_mean": float(pi_src.mean()),
                "pi_tgt_mean": float(pi_tgt.mean()),
                "pvalue_plugin": pval_plugin,
            },
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        X_src: np.ndarray,
        y_src: np.ndarray,
        X_tgt: np.ndarray,
        y_tgt: Optional[np.ndarray] = None,
    ) -> Tuple[ShiftTestResult, Optional[ShiftTestResult]]:
        """Run the aggregate covariate-shift and outcome-shift tests on the
        test split of each domain.

        Parameters
        ----------
        X_src, y_src : ndarray
            Source-domain test covariates and labels.
        X_tgt : ndarray
            Target-domain test covariates.
        y_tgt : ndarray, optional
            Target-domain test labels. Required for the outcome-shift test,
            which is skipped (returned as ``None``) when not given.

        Returns
        -------
        (covariate_result, outcome_result) : ShiftTestResult, ShiftTestResult or None
        """
        X_src, y_src = np.asarray(X_src, dtype=float), np.asarray(y_src, dtype=float)
        X_tgt = np.asarray(X_tgt, dtype=float)

        rng = np.random.default_rng(self.random_state)

        source_loss = self._loss_func(X_src, y_src)

        outcome_result = None
        if y_tgt is not None:
            y_tgt = np.asarray(y_tgt, dtype=float)
            # ---- nuisance estimates on the test split (full features; the
            # outcome test always uses these, the covariate test recomputes its
            # own optionally feature-filtered Z0/π internally) ----
            Z0_src = self._compute_Z(X_src, X_src, self.source_error_model)
            Z0_tgt = self._compute_Z(X_tgt, X_tgt, self.source_error_model)
            target_loss = self._loss_func(X_tgt, y_tgt)

            density_available = self.density_model is not None
            pi_src = self._density_ratio(X_src) if density_available else None

            outcome_result = self._run_outcome_test(
                X_src, X_tgt, Z0_src, Z0_tgt, source_loss, target_loss, pi_src, rng
            )
        covariate_result = self._run_covariate_test(X_src, X_tgt, source_loss, rng)

        return covariate_result, outcome_result

    # ------------------------------------------------------------------
    # Feature attribution: "which variables explain the shift", without
    # DetectorTestExplainer's combinatorial --combos search.
    # ------------------------------------------------------------------

    @staticmethod
    def _get_shap_module():
        try:
            import shap
        except ImportError as e:
            raise ImportError(
                "explain_covariate_shift()/explain_outcome_shift() require the "
                "`shap` package (`pip install shap`)."
            ) from e
        return shap

    def _subsample(self, X: np.ndarray, n: int) -> np.ndarray:
        if len(X) <= n:
            return X
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=n, replace=False)
        return X[idx]

    def _feature_names(self, n_feats: int, mask: Optional[np.ndarray] = None) -> list:
        """Feature names for `n_feats` columns, optionally restricted to `mask`
        (indices into the *original*, unfiltered feature set)."""
        if self.feature_names is not None and len(self.feature_names) >= n_feats:
            names = list(self.feature_names)
        else:
            names = [f"x{i}" for i in range(len(mask) if mask is not None else n_feats)]
        if mask is not None:
            names = [name for name, keep in zip(names, mask) if keep]
        return names

    def _explain_X_pool(self, X_src: Optional[np.ndarray] = None, X_tgt: Optional[np.ndarray] = None):
        """(X_src, X_tgt) to explain on: the given arrays if both are given,
        else the train data used at construction time."""
        if X_src is not None and X_tgt is not None:
            return np.asarray(X_src, dtype=float), np.asarray(X_tgt, dtype=float)
        if self._src_train is None or self._tgt_train is None:
            raise ValueError(
                "No X_src/X_tgt were given and no source_train_data/target_train_data "
                "is available to explain on. Pass X_src=..., X_tgt=... explicitly."
            )
        return self._src_train[0], self._tgt_train[0]

    def explain_covariate_shift(
        self,
        X_src: Optional[np.ndarray] = None,
        X_tgt: Optional[np.ndarray] = None,
        filtered: bool = False,
        max_samples: int = 200,
        background_samples: int = 100,
    ) -> pd.DataFrame:
        """Per-feature attribution for covariate shift, without needing a
        --combos search.

        The density-ratio model π(x) is itself a source-vs-target classifier
        (predicting P(target | x)); its SHAP values at a pool of source +
        target points directly say which features it relies on to tell the
        two domains apart — i.e. which features' distributions shifted.

        Parameters
        ----------
        X_src, X_tgt : ndarray, optional
            Source / target points to explain on. Defaults to the train data
            used at construction.
        filtered : bool, default False
            Use `density_model_filter`/`correlated_features` (the model
            actually driving the covariate *test* when
            `filter_independent_features` was used) instead of the main
            `density_model`.
        max_samples : int
            Number of (source + target, pooled) points to compute SHAP
            values for.
        background_samples : int
            Number of points used as the SHAP background/reference distribution.

        Returns
        -------
        pd.DataFrame with columns [feature, mean_abs_shap, mean_shap],
        sorted by mean_abs_shap descending. mean_abs_shap is the global
        importance; mean_shap's sign says whether the feature pushes
        predictions toward "target" (positive) or "source" (negative) on
        average.
        """
        shap = self._get_shap_module()
        model = self.density_model_filter if filtered else self.density_model
        if model is None:
            raise RuntimeError(
                "No density model available "
                f"({'density_model_filter' if filtered else 'density_model'} is None)."
            )

        X_src, X_tgt = self._explain_X_pool(X_src, X_tgt)
        mask = self.correlated_features if (filtered and self.correlated_features is not None) else None
        if mask is not None:
            X_src, X_tgt = X_src[:, mask], X_tgt[:, mask]
        X_pool = np.vstack([X_src, X_tgt])

        background = self._subsample(X_pool, background_samples)
        X_sample = self._subsample(X_pool, max_samples)

        explainer = shap.Explainer(lambda X: model.predict_proba(X)[:, 1], background)
        shap_values = explainer(X_sample).values

        return pd.DataFrame({
            "feature": self._feature_names(X_pool.shape[1], mask),
            "mean_abs_shap": np.abs(shap_values).mean(axis=0),
            "mean_shap": shap_values.mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def explain_outcome_shift(
        self,
        X_src: Optional[np.ndarray] = None,
        X_tgt: Optional[np.ndarray] = None,
        max_samples: int = 200,
        background_samples: int = 100,
    ) -> pd.DataFrame:
        """Per-feature attribution for outcome/concept shift ("explanation
        shift"), without needing a --combos search.

        source_error_model (μ_0) and target_error_model (μ_1) are two
        models of the *same* prediction task fit on different domains.
        Rather than regressing the noisy realized-loss residual on X (what
        the aggregate outcome-shift detector does, and which never uses
        μ_1 at all), this explains each model's own predictions with SHAP
        at the same points x, then takes the per-feature *difference*
        Δφ_j(x) = φ_1,j(x) - φ_0,j(x). Because SHAP values are additive
        (Σ_j φ_j(x) ≈ model's prediction), Σ_j Δφ_j(x) ≈ μ_1(x) - μ_0(x)
        exactly — i.e. an honest, additive decomposition of *why* the two
        models disagree at each point, feature by feature.

        Parameters
        ----------
        X_src, X_tgt, max_samples, background_samples : see `explain_covariate_shift`.

        Returns
        -------
        pd.DataFrame with columns [feature, mean_abs_shap_diff, mean_shap_diff],
        sorted by mean_abs_shap_diff descending.
        """
        shap = self._get_shap_module()
        if self.target_error_model is None:
            self.target_error_model = self._resolve_error_model(None, self._tgt_train, "target")

        X_src, X_tgt = self._explain_X_pool(X_src, X_tgt)
        X_pool = np.vstack([X_src, X_tgt])

        background = self._subsample(X_pool, background_samples)
        X_sample = self._subsample(X_pool, max_samples)

        explainer_0 = shap.Explainer(lambda X: self.source_error_model.predict_proba(X)[:, 1], background)
        explainer_1 = shap.Explainer(lambda X: self.target_error_model.predict_proba(X)[:, 1], background)
        delta = explainer_1(X_sample).values - explainer_0(X_sample).values

        return pd.DataFrame({
            "feature": self._feature_names(X_pool.shape[1]),
            "mean_abs_shap_diff": np.abs(delta).mean(axis=0),
            "mean_shap_diff": delta.mean(axis=0),
        }).sort_values("mean_abs_shap_diff", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# MED3pa explanation method
# ---------------------------------------------------------------------------

class ShiftExplainer(CovariateConceptShiftDetector):
    """
    Aggregate covariate-shift and outcome-shift tests of Singh et al. (ICML 2025) as a MED3pa explanation method.

    The reference dataset is the source domain and the target dataset is the target domain. Each call tests
    whether the patients of a profile (e.g. males over 50) experience a covariate or outcome shift: the source
    patients of the profile are selected in the reference dataset with ``profile_path``, and the target patients of
    the profile are ``x_q`` (and ``y_q``).

    Both domains are split once into train/test parts, globally (by hashing each patient's features, so the split
    of a profile is exactly the restriction of the global split to its patients):

    * the **error models** (source and target outcome models μ_0, μ_1) are **global**: fitted once, on the train
      part of the whole reference / target datasets, and reused for every profile;
    * the **density-ratio model** is either global (``density_model_scope="global"``, fitted once on the train
      parts of the whole datasets) or refitted on the train parts of each profile (``"profile"``). In both cases
      the ratio is rescaled by the profile's train sizes, which turns the global ratio p_1(x)/p_0(x) into the
      profile's conditional ratio p_1(x|profile)/p_0(x|profile);
    * the **detectors** (covariate-shift (ω, λ) detectors and outcome-shift detector) are always fitted on the
      train parts of the profile, since they describe the shift within the profile;
    * the tests are evaluated on the test parts of the profile.

    ``f_model`` is the model being explained: unless ``loss_func`` is given, the loss is its binary cross-entropy.
    The outcome-shift test requires the target labels ``y_q``, and is skipped when they are not given. The target
    error model is only used by the outcome-shift attribution (``compute_attributions=True``), so it is only fitted
    when needed.

    Args:
        target_dataset (MaskedDataset): The whole target dataset, used to fit the global target models. Set by
            ``Explainer.set_target_dataset`` in MED3pa experiments. Its true labels are only needed for the target
            error model.
        density_model_scope (str): ``"global"`` (default) or ``"profile"``, see above.
        train_fraction (float): Fraction of each domain used to fit the models and detectors.
        min_samples_size (int): Minimum number of samples required in each train/test part of each domain of the
            profile.
        significance_level, tolerance, prevalence, n_bootstrap, one_sided_test, test_greater:
            See :class:`AggregateShiftTester`.
        compute_attributions (bool): Also return SHAP-based per-feature attributions of each shift (requires
            ``shap``).
        loss_func (Callable): Per-sample loss ``loss_func(X, y)`` of the model being explained.
        random_state (int): Seed of the train/test split, the internally-fitted models and the bootstrap.
        tester_kwargs (dict): Extra keyword arguments passed to :class:`AggregateShiftTester` (e.g. ``num_omega``,
            ``num_lambda``, ``density_classifier``, ``is_oracle``, ``filter_independent_features``,
            ``independence_test``). ``density_classifier`` and ``is_oracle`` are also used for the global models.
            With ``filter_independent_features=True``, the filtered source outcome and density models are fitted
            per profile.
        reference_dataset (MaskedDataset): The reference (source) dataset.
        f_model (Any): The model being explained, with a ``predict_proba`` method.
    """
    uses_target_labels = True
    # fitted lazily from the reference/target datasets and settings, which are already in the checkpoint key
    _cache_exclude = ("_global_models",)
    __density_model_scopes = ("global", "profile")

    def __init__(self, *,
                 target_dataset: Optional[MaskedDataset] = None, density_model_scope: str = "global",
                 train_fraction: float = 0.5, min_samples_size: int = 20, significance_level: float = 0.05,
                 tolerance: float = 0.0, prevalence: float = 0.0, n_bootstrap: int = 10000,
                 one_sided_test: bool = True, test_greater: bool = True, compute_attributions: bool = False,
                 loss_func: Optional[Callable] = None, random_state: Optional[int] = 42,
                 tester_kwargs: Optional[dict] = None,
                 reference_dataset: Optional[MaskedDataset] = None, f_model: Optional[Any] = None):
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive).")
        if density_model_scope not in self.__density_model_scopes:
            raise ValueError(f"density_model_scope must be one of {self.__density_model_scopes}, "
                             f"got {density_model_scope!r}.")
        self._reference_dataset = reference_dataset
        self._f_model = f_model
        self._target_dataset = target_dataset
        self.__density_model_scope = density_model_scope
        self.__train_fraction = train_fraction
        self.__min_samples_size = min_samples_size
        self.__significance_level = significance_level
        self.__tolerance = tolerance
        self.__prevalence = prevalence
        self.__n_bootstrap = n_bootstrap
        self.__one_sided_test = one_sided_test
        self.__test_greater = test_greater
        self.__compute_attributions = compute_attributions
        self.__loss_func = loss_func
        self.__random_state = random_state
        self.__tester_kwargs = tester_kwargs if tester_kwargs is not None else {}
        self._global_models = {}

    def set_context(self, reference_dataset: MaskedDataset, f_model: Any):
        super().set_context(reference_dataset=reference_dataset, f_model=f_model)
        self._global_models = {}

    def set_target_dataset(self, target_dataset: MaskedDataset):
        self._target_dataset = target_dataset
        self._global_models = {}

    @property
    def target_dataset(self) -> Optional[MaskedDataset]:
        return self._target_dataset

    def _call__impl(self, x_q: np.ndarray, profile_path: Optional[str] = None, y_q: Optional[np.ndarray] = None):
        if profile_path is None:
            profile_path = ["*"]

        assert self.context_is_set(), ("Reference set and base model must be set before applying the "
                                       "ShiftExplainer test.")
        print(f"Executing ShiftExplainer test in profile: {profile_path}")

        # filter out only profile in reference
        reference_mask = self._reference_dataset.mask_by_profile(path=profile_path)
        x_p = self._reference_dataset.get_observations()[reference_mask]
        y_p = self._reference_dataset.get_true_labels()[reference_mask]
        x_q = np.asarray(x_q)
        if y_q is not None:
            y_q = np.asarray(y_q)
            if len(y_q) != len(x_q):
                raise ValueError(f"x_q and y_q must have the same length, got {len(x_q)} and {len(y_q)}.")

        # restriction of the global train/test split to the profile
        p_train = self.__split(x_p)
        q_train = self.__split(x_q)

        # Each train/test part of each domain must contain enough samples
        part_sizes = {"reference": min(p_train.sum(), (~p_train).sum()), "test": min(q_train.sum(), (~q_train).sum())}
        smallest = min(part_sizes, key=part_sizes.get)
        if part_sizes[smallest] < self.__min_samples_size:
            return f"Not enough samples in {smallest}: {part_sizes[smallest]}"

        global_models = self.__get_global_models()
        tester = AggregateShiftTester(
            source_error_model=global_models["source_error_model"],
            target_error_model=global_models.get("target_error_model"),
            density_model=global_models.get("density_model"),
            source_train_data=(x_p[p_train], y_p[p_train]),
            target_train_data=(x_q[q_train], y_q[q_train] if y_q is not None else None),
            feature_names=self._reference_dataset.features,
            ml_mdl=self._f_model,
            loss_func=self.__loss_func,
            tolerance=self.__tolerance,
            prevalence=self.__prevalence,
            significance_level=self.__significance_level,
            n_bootstrap=self.__n_bootstrap,
            one_sided_test=self.__one_sided_test,
            test_greater=self.__test_greater,
            random_state=self.__random_state,
            **self.__tester_kwargs,
        )
        covariate_result, outcome_result = tester(
            X_src=x_p[~p_train], y_src=y_p[~p_train],
            X_tgt=x_q[~q_train], y_tgt=y_q[~q_train] if y_q is not None else None,
        )

        results = {
            "decision": self.__decision(covariate_result, outcome_result),
            "covariate_shift": covariate_result.to_dict(),
            "outcome_shift": outcome_result.to_dict() if outcome_result is not None else None,
        }
        if self.__compute_attributions:
            results["covariate_attribution"] = tester.explain_covariate_shift(
                X_src=x_p[~p_train], X_tgt=x_q[~q_train]).to_dict(orient="records")
            if y_q is not None:
                results["outcome_attribution"] = tester.explain_outcome_shift(
                    X_src=x_p[~p_train], X_tgt=x_q[~q_train]).to_dict(orient="records")
        return results

    def __split(self, x: np.ndarray) -> np.ndarray:
        return _hash_split(x, self.__train_fraction, self.__random_state)

    def __get_global_models(self) -> dict:
        """Fit (once) the global models on the train parts of the whole reference and target datasets."""
        if self._global_models:
            return self._global_models

        if self._target_dataset is None and (self.__density_model_scope == "global" or self.__compute_attributions):
            raise ValueError("ShiftExplainer needs the whole target dataset to fit the global models: pass "
                             "target_dataset=... or call set_target_dataset(...) (Explainer.set_target_dataset).")
        is_oracle = self.__tester_kwargs.get("is_oracle", False)
        print("Fitting ShiftExplainer global models")

        x_p = self._reference_dataset.get_observations()
        y_p = self._reference_dataset.get_true_labels()
        p_train = self.__split(x_p)
        if len(np.unique(y_p[p_train])) < 2:
            raise ValueError("The train part of the reference dataset contains a single class: the source error "
                             "model cannot be fitted.")
        global_models = {"source_error_model": _fit_outcome_model(x_p[p_train], y_p[p_train], is_oracle=is_oracle)}

        if self._target_dataset is not None:
            x_q = self._target_dataset.get_observations()
            q_train = self.__split(x_q)
            if self.__density_model_scope == "global":
                template = _make_density_template(self.__tester_kwargs.get("density_classifier"),
                                                  self.__random_state)
                global_models["density_model"] = _fit_density_model(template, x_p[p_train], x_q[q_train])
            if self.__compute_attributions:
                y_q = self._target_dataset.get_true_labels()
                global_models["target_error_model"] = _fit_outcome_model(x_q[q_train], y_q[q_train],
                                                                         is_oracle=is_oracle)

        self._global_models = global_models
        return global_models

    @staticmethod
    def __decision(covariate_result: ShiftTestResult, outcome_result: Optional[ShiftTestResult]) -> str:
        outcome_detected = outcome_result is not None and outcome_result.detected
        if covariate_result.detected and outcome_detected:
            return "covariate and outcome shift"
        elif covariate_result.detected:
            return "covariate shift"
        elif outcome_detected:
            return "outcome shift"
        return "No shift"


# ---------------------------------------------------------------------------
# Quick smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Smoke test: generate synthetic source/target data with a known outcome shift
    (Setup 1a from the paper) and verify the outcome-shift test fires while
    the covariate-shift test does not.
    """
    rng = np.random.default_rng(0)
    n = 1000
    p = 5

    # ---- Source domain ----
    X_src = rng.normal(0, 1.5, (n, p))
    phi_src = 0.8 * X_src[:, 0] + 0.5 * X_src[:, 1]
    y_src = (rng.random(n) < 1 / (1 + np.exp(-phi_src))).astype(float)

    # ---- Target domain: outcome shift in subgroup {|x1| > 1.5} ----
    X_tgt = rng.normal(0, 1.5, (n, p))
    subgroup = np.abs(X_tgt[:, 0]) > 1.5
    phi_tgt_full = 0.8 * X_tgt[:, 0] + 0.5 * X_tgt[:, 1]
    phi_tgt_shifted = phi_tgt_full - 0.6 * X_tgt[:, 0] * subgroup
    y_tgt = (rng.random(n) < 1 / (1 + np.exp(-phi_tgt_shifted))).astype(float)

    cols = [f"x{i}" for i in range(p)]
    reference_dataset = MaskedDataset(observations=X_src, true_labels=y_src, column_labels=cols)
    target_dataset = MaskedDataset(observations=X_tgt, true_labels=y_tgt, column_labels=cols)

    # model being explained, trained on the source domain
    f_model = LogisticRegression(max_iter=1000)
    f_model.fit(X_src, y_src)

    for density_model_scope in ["global", "profile"]:
        explainer = ShiftExplainer(reference_dataset=reference_dataset, f_model=f_model,
                                   target_dataset=target_dataset, density_model_scope=density_model_scope,
                                   n_bootstrap=500)
        for profile_path in [["*"], ["x0 > 0"]]:
            print(f"=== Density model: {density_model_scope}, profile {profile_path} ===")
            target_mask = target_dataset.mask_by_profile(path=profile_path)
            # call the implementation directly: the checkpointer cache does not depend on the explainer's parameters
            print(explainer._call__impl(x_q=X_tgt[target_mask], y_q=y_tgt[target_mask], profile_path=profile_path))
            print()
