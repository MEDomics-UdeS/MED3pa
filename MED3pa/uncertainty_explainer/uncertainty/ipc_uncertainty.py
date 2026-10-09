"""
Explains whether the low confidence (high expected loss of the base model) of a MED3pa profile comes from epistemic
uncertainty: from the base model being more wrong than usual (reducible loss) where its kind of data is scarce, rather
than from inherently unpredictable outcomes (irreducible loss). Also gives the uncertainty of the confidence
estimates of the target and source IPC models (as used by ShiftExplainer).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from typing import Any, Dict, Optional

from MED3pa.datasets import MaskedDataset
from MED3pa.uncertainty_explainer.abstract_methods import UncertaintyEstimator
from MED3pa.uncertainty_explainer.error_models import fit_confidence_model
from MED3pa.uncertainty_explainer.shift.shift_explainer import (_fit_density_model, _hash_split,
                                                                _make_density_template)

# IPC regressors trained without sample weights: their prediction is E[c | x], invertible with a class weight ratio 1
_UNWEIGHTED_REGRESSORS = ("RandomForestRegressor", "DecisionTreeRegressor")


class IPCUncertainty(UncertaintyEstimator):
    """
    Explains the low confidence of a profile, for its target samples x_q and for the samples of the same profile in the
    reference dataset:

    - loss split: the expected loss 1 - E[c | x] of the base model predictions, split into an irreducible
      (aleatoric) part, the minimum expected loss achievable at x (min over q of the loss of predicting q when
      P(y = 1 | x) = pi(x)), and a reducible (epistemic, model-induced) part, the rest. pi(x) and E[c | x] come from
      the IPC model (debiased for a BalancedEnsembleRandomForestRegressor, see ``predict_debiased``): the target IPC
      model for the target samples, the source IPC model for the reference samples. The reducible loss also contains
      the base model's operating choices (e.g. class weighting), present in every profile, so it is compared with the
      one of all the samples (root profile): ``excess_reducible`` and ``excess_ratio``.
    - data density: how much the base model's kind of data (the reference dataset) is scarce in the profile, with the
      density ratio p_target(x) / p_reference(x) (relative under-representation; the global density model of the
      ShiftExplainer of the same Explainer when available) and the kNN distance to the reference (mean distance to
      its ``n_neighbors`` nearest reference samples on standardized features, relative to the median one of the
      reference samples: absolute sparsity).
    - verdict: epistemic when the reducible loss exceeds the one of all the samples by ``excess_threshold`` (ratio)
      where the base model's kind of data is under-represented (density ratio >= ``density_threshold``). The kNN
      distance is reported but not used: it also flags the tails of any feature, even irrelevant ones.
    - variance indicator: an epistemic indicator using only data of the training distribution (no target data): the
      variance, over ``variance_n_models`` depth-limited trees (``variance_max_depth``) predicting the outcome, each
      fitted on a bootstrap resample of the reference samples used to train the source IPC model (the classical
      bootstrap epistemic uncertainty of a base-model-like learner), on the samples of the profile relative to its
      mean on the held-out reference samples (``ratio``; sparse in the training data when >=
      ``variance_threshold``). Depth-limited trees keep fewer samples in their leaves where the data is sparse, so
      their variance grows there (unlike trees grown to a minimum leaf size, such as the IPC submodels). It measures
      the absolute sparsity of the training data (e.g. also the tails of a feature), not the relative
      under-representation of the density ratio, which needs the target data, and it also grows with the outcome
      noise.
    - confidence estimate uncertainty: the ``predict_uncertainty`` of the target IPC model (its submodel variance is
      the reliability of the confidence estimate) and of the source IPC model, with the mean predicted confidence and
      its aleatoric, epistemic and total variances.

    The source IPC model is the one given to ``set_error_model``, else the one of the ShiftExplainer of the same
    Explainer, else a new instance of the same IPC class trained on the ``train_fraction`` part of the reference
    dataset. The reference samples used to train it are left out of the reference samples of the profile (when
    known). The target dataset (``set_target_dataset``, set by the Explainer in MED3pa experiments) is needed for the
    root profile of the target samples and to fit the density model.
    """
    # derived from the settings and datasets, which are in the checkpoint key
    _cache_exclude = ("_source_error_model_provider", "_density_model", "_density_train_ratio", "_knn", "_root",
                      "_variance_models")

    def __init__(self, *,
                 min_samples_size: int = 1, train_fraction: float = 0.5, random_state: Optional[int] = 42,
                 n_neighbors: int = 20, density_threshold: float = 1.5, knn_threshold: float = 1.2,
                 excess_threshold: float = 1.2, variance_threshold: float = 1.2, variance_n_models: int = 30,
                 variance_max_depth: int = 6, q_grid_size: int = 201,
                 target_dataset: Optional[MaskedDataset] = None,
                 reference_dataset: Optional[MaskedDataset] = None, f_model: Optional[Any] = None):
        """
        Args:
            min_samples_size (int): Minimum number of samples of the profile to explain it.
            train_fraction (float): Fraction of the reference (and target) dataset used to train the source IPC model
                and the density model when they are fitted here (the same default split as ShiftExplainer).
            random_state (int): Seed of the splits and of the density model.
            n_neighbors (int): Number of nearest reference samples of the kNN distance.
            density_threshold (float): Density ratio from which the profile is under-represented in the reference.
            knn_threshold (float): Relative kNN distance from which the profile is sparse in the reference (reported
                in ``data_density``, not used by the verdict).
            excess_threshold (float): Ratio of the profile's reducible loss to the one of all the samples from which
                the base model is more wrong than usual in the profile.
            variance_threshold (float): Ratio of the variance of the bootstrap trees on the profile to its mean on the
                held-out reference samples from which the profile is sparse in the training data.
            variance_n_models (int): Number of bootstrap trees of the variance indicator.
            variance_max_depth (int): Maximum depth of the bootstrap trees of the variance indicator.
            q_grid_size (int): Number of predictions q of the grid of the minimum expected loss.
            target_dataset (MaskedDataset): The whole target dataset.
            reference_dataset (MaskedDataset): The reference (source) dataset.
            f_model (Any): The model being explained, with a ``predict_proba`` method.
        """
        if not 0.0 < train_fraction < 1.0:
            raise ValueError("train_fraction must be between 0 and 1 (exclusive).")
        self._reference_dataset = reference_dataset
        self._f_model = f_model
        self._target_dataset = target_dataset
        self.__min_samples_size = min_samples_size
        self.__train_fraction = train_fraction
        self.__random_state = random_state
        self.__n_neighbors = n_neighbors
        self.__density_threshold = density_threshold
        self.__knn_threshold = knn_threshold
        self.__excess_threshold = excess_threshold
        self.__variance_threshold = variance_threshold
        self.__variance_n_models = variance_n_models
        self.__variance_max_depth = variance_max_depth
        self.__q_grid = np.linspace(0.0, 1.0, q_grid_size)
        self._error_model = None
        self._confidence_calc = None
        self._source_error_model = None
        self._source_error_model_provider = None
        self._fitted_source_error_model = None
        self._source_training_mask = None
        self.__reset_fitted()

    def __reset_fitted(self, source: bool = False):
        if source:
            self._fitted_source_error_model = None
            self._source_training_mask = None
        self._density_model = None
        self._density_train_ratio = None
        self._knn = None
        self._variance_models = None
        self._root = {}

    def set_context(self, reference_dataset: MaskedDataset, f_model: Any):
        super().set_context(reference_dataset=reference_dataset, f_model=f_model)
        self.__reset_fitted(source=True)

    def set_target_dataset(self, target_dataset: MaskedDataset):
        """Sets the whole target dataset, for the root profile of the target samples and the density model."""
        self._target_dataset = target_dataset
        self.__reset_fitted()

    def set_error_model(self, error_model: Any, confidence_calc: Any = None, source_error_model: Optional[Any] = None):
        """
        Sets the trained IPC model (the target IPC model), the confidence calculator giving the confidence it
        predicts and optionally an already-trained source IPC model.
        """
        self._error_model = error_model
        self._confidence_calc = confidence_calc
        self._source_error_model = source_error_model
        self.__reset_fitted(source=True)

    def set_source_error_model_provider(self, provider: Any):
        """
        Sets the method providing the source IPC model (with ``get_source_error_model`` and
        ``get_source_error_model_training_mask``, e.g. ShiftExplainer), to share it instead of training another one,
        and its global density ratio (``get_density_ratio``) when available.
        """
        self._source_error_model_provider = provider
        self.__reset_fitted(source=True)

    def get_source_error_model(self) -> Optional[Any]:
        """
        Source IPC model: the given one, else the one of the provider, else the one trained here on the train part
        of the reference dataset (trained if it was not already). None when no error model is set, or when it cannot
        be trained (no confidence calculator, single class in the reference dataset).
        """
        if self._error_model is None:
            return None
        if self._source_error_model is not None:
            return self._source_error_model
        if self._fitted_source_error_model is None:
            x_p = self._reference_dataset.get_observations()
            if self._source_error_model_provider is not None:
                self._fitted_source_error_model = self._source_error_model_provider.get_source_error_model()
                self._source_training_mask = self._source_error_model_provider.get_source_error_model_training_mask(
                    x_p)
            else:
                y_p = self._reference_dataset.get_true_labels()
                p_train = _hash_split(x_p, self.__train_fraction, self.__random_state)
                if self._confidence_calc is None or len(np.unique(y_p[p_train])) < 2:
                    return None
                confidence_p = self._confidence_calc.calculate_confidence(
                    x_p[p_train], self._f_model.predict_proba(x_p[p_train])[:, 1], y_p[p_train])
                self._fitted_source_error_model = fit_confidence_model(self._error_model, self._f_model,
                                                                       x_p[p_train], y_p[p_train], confidence_p)
                self._source_training_mask = p_train
        return self._fitted_source_error_model

    def _checkpoint_id(self, **call_kwargs) -> str:
        # the source IPC model (possibly from the provider, not in the key) is resolved before computing the key
        self.get_source_error_model()
        return super()._checkpoint_id(**call_kwargs)

    # ------------------------------------------------------------------
    # Confidence estimate uncertainty
    # ------------------------------------------------------------------

    @staticmethod
    def _supports_uncertainty(model: Any) -> bool:
        # the regressor of an IPCModel (IPCModel.predict_uncertainty exists but raises for unsupported regressors)
        return hasattr(getattr(model, 'model', model), 'predict_uncertainty')

    @staticmethod
    def _uncertainty_summary(model: Any, x: np.ndarray) -> Dict[str, float]:
        uncertainty = model.predict_uncertainty(x)
        summary = {'mean_confidence': float(np.mean(uncertainty['mean']))}
        summary.update({key: float(np.mean(uncertainty[key])) for key in ('aleatoric', 'epistemic', 'total')})
        summary['epistemic_share'] = summary['epistemic'] / summary['total'] if summary['total'] > 0 else None
        return summary

    # ------------------------------------------------------------------
    # Loss split
    # ------------------------------------------------------------------

    def _possible_confidences(self, x: np.ndarray):
        """Confidence of the base model predictions if y = 0 and if y = 1."""
        p = self._f_model.predict_proba(x)[:, 1]
        return (self._confidence_calc.calculate_confidence(x, p, np.zeros_like(p)),
                self._confidence_calc.calculate_confidence(x, p, np.ones_like(p)))

    def _confidence_and_risk(self, model: Any, x: np.ndarray):
        """E[c | x] and pi(x) = P(y = 1 | x) from an IPC model, or None when it cannot give them."""
        c0, c1 = self._possible_confidences(x)
        if hasattr(model, 'debiasing_available') and model.debiasing_available():
            return model.model.predict_debiased(x, c0, c1, return_risk=True)
        if getattr(model, 'model_name', None) in _UNWEIGHTED_REGRESSORS:
            # E[c | x] = pi c1 + (1 - pi) c0
            confidence = np.clip(np.asarray(model.predict(x), dtype=float), np.minimum(c0, c1), np.maximum(c0, c1))
            with np.errstate(divide='ignore', invalid='ignore'):
                risk = np.where(np.abs(c0 - c1) > 1e-9, (c0 - confidence) / (c0 - c1), 0.5)
            return confidence, np.clip(risk, 0.0, 1.0)
        return None

    def _irreducible_loss(self, x: np.ndarray, risk: np.ndarray) -> np.ndarray:
        """Minimum over q of the expected loss of predicting q when P(y = 1 | x) = risk."""
        minimum = np.full(len(x), np.inf)
        ones, zeros = np.ones(len(x)), np.zeros(len(x))
        for q in self.__q_grid:
            q_x = np.full(len(x), q)
            loss = (risk * (1 - self._confidence_calc.calculate_confidence(x, q_x, ones))
                    + (1 - risk) * (1 - self._confidence_calc.calculate_confidence(x, q_x, zeros)))
            minimum = np.minimum(minimum, loss)
        return minimum

    def _loss_split(self, model: Any, x: np.ndarray) -> Optional[Dict[str, float]]:
        """Mean expected loss of the base model predictions on x, and its irreducible and reducible parts."""
        if model is None or len(x) == 0 or self._confidence_calc is None:
            return None
        estimate = self._confidence_and_risk(model, x)
        if estimate is None:
            return None
        confidence, risk = estimate
        total = 1.0 - confidence
        irreducible = self._irreducible_loss(x, risk)
        reducible = np.maximum(total - irreducible, 0.0)
        split = {'total': float(total.mean()), 'irreducible': float(irreducible.mean()),
                 'reducible': float(reducible.mean())}
        split['reducible_share'] = split['reducible'] / split['total'] if split['total'] > 0 else None
        return split

    def _root_reducible(self, domain: str, model: Any) -> Optional[float]:
        """Mean reducible loss of all the target samples (target dataset) or all the held-out reference samples."""
        if domain not in self._root:
            if domain == 'target':
                x = self._target_dataset.get_observations() if self._target_dataset is not None else None
            else:
                x = self._reference_dataset.get_observations()
                if self._source_training_mask is not None:
                    x = x[~self._source_training_mask]
            split = self._loss_split(model, x) if x is not None else None
            self._root[domain] = split['reducible'] if split is not None else None
        return self._root[domain]

    # ------------------------------------------------------------------
    # Data density
    # ------------------------------------------------------------------

    def _density_ratio(self, x: np.ndarray) -> Optional[np.ndarray]:
        """p_target(x) / p_reference(x): the provider's global density model, else one fitted here."""
        if self._source_error_model_provider is not None and hasattr(self._source_error_model_provider,
                                                                      'get_density_ratio'):
            ratio = self._source_error_model_provider.get_density_ratio(x)
            if ratio is not None:
                return ratio
        if self._target_dataset is None:
            return None
        if self._density_model is None:
            x_p, x_q = self._reference_dataset.get_observations(), self._target_dataset.get_observations()
            p_train = _hash_split(x_p, self.__train_fraction, self.__random_state)
            q_train = _hash_split(x_q, self.__train_fraction, self.__random_state)
            self._density_model = _fit_density_model(_make_density_template(None, self.__random_state),
                                                     x_p[p_train], x_q[q_train])
            self._density_train_ratio = float(p_train.sum() / q_train.sum())
        proba = np.clip(self._density_model.predict_proba(np.asarray(x, dtype=float))[:, 1], 1e-3, 1 - 1e-3)
        return proba / (1 - proba) * self._density_train_ratio

    def _knn_distance(self, x: np.ndarray, exclude_self: bool = False) -> np.ndarray:
        """Mean distance to the n_neighbors nearest reference samples, relative to the median one of the reference
        samples (standardized features). `exclude_self` for reference samples."""
        if self._knn is None:
            x_p = self._reference_dataset.get_observations()
            scaler = StandardScaler().fit(x_p)
            n_neighbors = min(self.__n_neighbors, len(x_p) - 1)
            nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(scaler.transform(x_p))
            reference_distance = nn.kneighbors(scaler.transform(x_p))[0][:, 1:].mean(axis=1)
            self._knn = (scaler, nn, n_neighbors, float(np.median(reference_distance)))
        scaler, nn, n_neighbors, median_distance = self._knn
        distances = nn.kneighbors(scaler.transform(x), n_neighbors=n_neighbors + int(exclude_self))[0]
        if exclude_self:
            distances = distances[:, 1:]
        return distances.mean(axis=1) / median_distance

    def _reference_split(self) -> np.ndarray:
        """Train mask of the reference samples: the ones of the source IPC model when known, else a split."""
        if self._source_training_mask is not None:
            return self._source_training_mask
        return _hash_split(self._reference_dataset.get_observations(), self.__train_fraction, self.__random_state)

    def _bootstrap_variance(self, x: np.ndarray) -> np.ndarray:
        """Variance of the predicted outcome probability over the bootstrap trees fitted on the reference."""
        if self._variance_models is None:
            train = self._reference_split()
            x_p = self._reference_dataset.get_observations()[train]
            y_p = self._reference_dataset.get_true_labels()[train]
            rng = np.random.default_rng(self.__random_state)
            models = []
            for _ in range(self.__variance_n_models):
                idx = rng.integers(0, len(x_p), len(x_p))
                models.append(DecisionTreeClassifier(max_depth=self.__variance_max_depth,
                                                     random_state=int(rng.integers(np.iinfo(np.int32).max)))
                              .fit(x_p[idx], y_p[idx]))
            self._variance_models = models
        probabilities = []
        for model in self._variance_models:
            proba = model.predict_proba(x)
            probabilities.append(proba[:, list(model.classes_).index(1)] if 1 in model.classes_
                                 else np.zeros(len(x)))
        return np.var(np.array(probabilities), axis=0, ddof=1)

    def _variance_indicator(self, x_q: np.ndarray, x_reference: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        """Variance of the bootstrap trees on the profile relative to its mean on the held-out reference samples."""
        if 'variance_baseline' not in self._root:
            held_out = ~self._reference_split()
            self._root['variance_baseline'] = (
                float(np.mean(self._bootstrap_variance(self._reference_dataset.get_observations()[held_out])))
                if held_out.any() else None)
        baseline = self._root['variance_baseline']
        if not baseline:
            return None
        variance = float(np.mean(self._bootstrap_variance(x_q)))
        ratio = variance / baseline
        ratio_reference = (float(np.mean(self._bootstrap_variance(x_reference))) / baseline
                           if x_reference is not None and len(x_reference) > 0 else None)
        return {'variance': variance, 'training_baseline': baseline, 'ratio': ratio,
                'ratio_reference': ratio_reference,
                'variance_verdict': ("sparse in the training data (higher bootstrap variance)"
                                     if ratio >= self.__variance_threshold
                                     else "not sparse in the training data")}

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def _verdict(self, split: Optional[Dict[str, float]], excess_ratio: Optional[float],
                 density_ratio: Optional[float]) -> str:
        if split is None or excess_ratio is None:
            return "loss split not available"
        under_represented = density_ratio is not None and density_ratio >= self.__density_threshold
        more_wrong = excess_ratio >= self.__excess_threshold
        if more_wrong and under_represented:
            return "epistemic: excess model-induced loss where the reference data is scarce"
        if more_wrong:
            return "excess model-induced loss, not explained by data scarcity"
        if under_represented:
            return "reference data scarce, without excess model-induced loss"
        return "no excess model-induced loss"

    def _call__impl(self, x_q: np.ndarray, profile_path: Optional[str] = None):
        if profile_path is None:
            profile_path = ["*"]

        assert self.context_is_set(), "Reference set and base model must be set before applying IPCUncertainty."
        print(f"Executing IPCUncertainty in profile: {profile_path}")

        if self._error_model is None:
            return "Uncertainty estimation not implemented: no IPC model set (Explainer.set_error_model)."
        if len(x_q) < self.__min_samples_size:
            return f"Not enough samples in test: {len(x_q)}"
        x_q = np.asarray(x_q, dtype=float)

        # reference samples of the profile, without the ones used to train the source IPC model
        source_error_model = self.get_source_error_model()
        reference_mask = self._reference_dataset.mask_by_profile(path=profile_path)
        if self._source_training_mask is not None:
            reference_mask = reference_mask & ~self._source_training_mask
        x_reference = self._reference_dataset.get_observations()[reference_mask]
        enough_reference = len(x_reference) >= self.__min_samples_size

        # 1. loss split, compared with all the samples of each domain
        loss_split = {}
        for domain, model, x in (("target", self._error_model, x_q),
                                 ("reference", source_error_model, x_reference if enough_reference else None)):
            split = self._loss_split(model, x) if x is not None else None
            if split is not None:
                root = self._root_reducible(domain, model)
                split['root_reducible'] = root
                split['excess_reducible'] = split['reducible'] - root if root is not None else None
                split['excess_ratio'] = split['reducible'] / root if root else None
            loss_split[domain] = split

        # 2. data density of the reference (the base model's kind of data)
        density = self._density_ratio(x_q)
        density_reference = self._density_ratio(x_reference) if enough_reference else None
        data_density = {
            'density_ratio': {'target': float(np.mean(density)) if density is not None else None,
                              'reference': float(np.mean(density_reference))
                              if density_reference is not None else None},
            'knn_distance': {'target': float(np.mean(self._knn_distance(x_q))),
                             'reference': float(np.mean(self._knn_distance(x_reference, exclude_self=True)))
                             if enough_reference else None},
        }

        # 3. confidence estimate uncertainty of the IPC models
        def summarize(model: Any, name: str) -> Dict[str, Any] | str:
            if model is None:
                return f"{name} IPC model not available (it could not be trained on the reference set)."
            if not self._supports_uncertainty(model):
                return (f"Uncertainty estimation not implemented for the {name.lower()} IPC model "
                        f"{getattr(model, 'model_name', type(model).__name__)}.")
            return {'target': self._uncertainty_summary(model, x_q),
                    'reference': self._uncertainty_summary(model, x_reference) if enough_reference else None}

        results = {'target_ipc': summarize(self._error_model, "Target"),
                   'source_ipc': summarize(source_error_model, "Source")}

        target_split = loss_split['target']
        verdict = self._verdict(target_split, target_split['excess_ratio'] if target_split else None,
                                data_density['density_ratio']['target'])
        data_density['knn_sparse'] = bool(data_density['knn_distance']['target'] >= self.__knn_threshold)
        variance = self._variance_indicator(x_q, x_reference if enough_reference else None)
        results.update({'verdict': verdict, 'loss_split': loss_split, 'data_density': data_density,
                        'variance_indicator': variance})

        parts = [verdict]
        if target_split is not None:
            ratio = (f" (x{target_split['excess_ratio']:.2f} vs all)"
                     if target_split['excess_ratio'] is not None else "")
            parts.append(f"loss {target_split['total']:.3f} = irreducible {target_split['irreducible']:.3f} + "
                         f"reducible {target_split['reducible']:.3f}{ratio}")
        density_text = (f"density ratio {data_density['density_ratio']['target']:.2f}, "
                        if data_density['density_ratio']['target'] is not None else "")
        parts.append(f"{density_text}kNN distance {data_density['knn_distance']['target']:.2f}")
        if variance is not None:
            parts.append(f"training-data sparsity (bootstrap variance) x{variance['ratio']:.2f}")
        if isinstance(results['target_ipc'], dict):
            parts.append(f"confidence estimate sd {np.sqrt(results['target_ipc']['target']['epistemic']):.3f}")
        results['decision'] = " | ".join(parts)
        results['n_target'] = len(x_q)
        results['n_reference'] = len(x_reference)
        return results
