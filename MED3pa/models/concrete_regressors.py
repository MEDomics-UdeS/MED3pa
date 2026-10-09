"""Similar to ``concrete_classifiers.py``, this module contains implementations of regression models like
RandomForestRegressor and DecisionTreeRegressor. It provides practical, ready-to-use models that comply with the
abstract definitions, making it easier to integrate and use these models in ``med3pa``."""

import itertools
import numpy as np
import warnings
import optuna
from copy import deepcopy
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from typing import Any, Dict, List, Optional, Tuple

from .abstract_models import RegressionModel
from .data_strategies import ToNumpyStrategy
from .regression_metrics import *


class RandomForestRegressorModel(RegressionModel):
    """
    A concrete implementation of the Model class for RandomForestRegressor models.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the RandomForestRegressorModel with a scikit-learn RandomForestRegressor.

        Args:
            params (dict): Parameters for initializing the RandomForestRegressor.
        """
        super().__init__()
        self.params = params
        self.model = RandomForestRegressor(**params)
        self.model_class = RandomForestRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None,
              y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None, **params) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            x_validation (np.ndarray, optional): observations for validation.
            y_validation (np.ndarray, optional): Labels for validation.
            training_parameters (dict, optional): Additional training parameters.

        Raises:
            ValueError: If the RandomForestRegressorModel has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The RandomForestRegressor has not been initialized.")

        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)

        if training_parameters:
            valid_param_sets = [set(self.model.get_params())]
            validated_params = self.validate_params(training_parameters, valid_param_sets)
            self.params.update(validated_params)
            self.model.set_params(**self.params)
        
        self.model.fit(np_X_train, np_y_train, **params)

        if x_validation is not None and y_validation is not None:
            self.evaluate(x_validation, y_validation, ['RMSE', 'MSE'], True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the model for the given input.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: Predictions made by the model.

        Raises:
            ValueError: If the RandomForestRegressorModel has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The RandomForestRegressorModel has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)


class EnsembleRandomForestRegressorModel(RegressionModel):
    """
    An ensemble model consisting of multiple RandomForestRegressorModel instances,
    with undersampling applied to the majority class.
    """

    def __init__(self, params: Dict[str, Any] = None,
                 base_model: RandomForestRegressorModel = RandomForestRegressorModel,
                 n_models: int = 10,
                 random_state: int = None,
                 **params_sklearn) -> None:
        """
        Initializes the EnsembleRandomForestRegressorModel with multiple RandomForestRegressor models.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for each model in the ensemble.
            base_model (RandomForestRegressorModel): A prototype instance of RandomForestRegressorModel.
            n_models (int): The number of RandomForestRegressorModel instances in the ensemble.
            random_state (int): A random_state can be set for reproducibility.
            **params_sklearn (Any) : Parameters for the sklearn model.
        """
        if params is None:
            params = {}
        if params_sklearn is not None:
            params.update(params_sklearn)
        super().__init__()
        self.params = params
        self.n_models = n_models
        self.models = []
        self.random_state = random_state
        for n_model in range(n_models):
            model = deepcopy(base_model(params))
            self.models.append(model)
        self.model = self

        self.fit = self.train

    def _undersample(self, x: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> tuple:
        """
        Applies undersampling to the majority class based on sample weights.
        Samples with lower sample weights are undersampled, while higher weighted samples are retained.
        """
        # Sort data by sample_weight
        sorted_indices = np.argsort(sample_weight)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        sample_weight_sorted = sample_weight[sorted_indices]

        # Identify the threshold to differentiate between lower and higher sample weights
        weight_threshold = np.median(sample_weight_sorted)

        # Split into "higher weight" and "lower weight" groups
        x_higher_weight = x_sorted[sample_weight_sorted >= weight_threshold]
        y_higher_weight = y_sorted[sample_weight_sorted >= weight_threshold]
        sample_weight_higher = sample_weight_sorted[sample_weight_sorted >= weight_threshold]

        x_lower_weight = x_sorted[sample_weight_sorted < weight_threshold]
        y_lower_weight = y_sorted[sample_weight_sorted < weight_threshold]
        sample_weight_lower = sample_weight_sorted[sample_weight_sorted < weight_threshold]

        # Undersample the lower-weight group
        if len(y_lower_weight) > len(y_higher_weight):
            x_lower_weight_resampled, y_lower_weight_resampled, sample_weight_lower_resampled = resample(
                x_lower_weight, y_lower_weight, sample_weight_lower,
                replace=False, n_samples=len(y_higher_weight),
                random_state=self.random_state
            )
        else:
            x_lower_weight_resampled, y_lower_weight_resampled, sample_weight_lower_resampled = (
                x_lower_weight, y_lower_weight, sample_weight_lower
            )

        # Combine higher-weight samples and resampled lower-weight samples
        x_resampled = np.vstack((x_higher_weight, x_lower_weight_resampled))
        y_resampled = np.hstack((y_higher_weight, y_lower_weight_resampled))
        sample_weight_resampled = np.hstack((sample_weight_higher, sample_weight_lower_resampled))

        return x_resampled, y_resampled, sample_weight_resampled

    def __sklearn_clone__(self):
        """
        Overwrites the sklearn clone function
        """
        new_instance = deepcopy(self)
        return new_instance

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None,
              y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None, **params) -> None:
        """
        Trains each model in the ensemble on a differently resampled dataset.
        """
        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)

        if training_parameters:
            self.params.update(training_parameters)
        if "sample_weight" not in params:
            raise ValueError("EnsembleRandomForestRegressorModel must be trained with a sample_weight parameter")
        sample_weight = params["sample_weight"]

        for model in self.models:
            # Resample the dataset for each model
            x_resampled, y_resampled, sample_weight = self._undersample(np_X_train, np_y_train, sample_weight)
            model.train(x_resampled, y_resampled, x_validation, y_validation, training_parameters, **params)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the ensemble model by averaging predictions from each model.
        """
        np_X, _ = self._ensure_numpy_arrays(X)

        predictions = np.zeros((self.n_models, len(np_X)))

        for i, model in enumerate(self.models):
            predictions[i] = model.predict(np_X)

        return np.mean(predictions, axis=0)

    def score(self, X, y, sample_weight=None):
        """ Taken from sklearn.base.py
        Return the coefficient of determination of the prediction.

        The coefficient of determination :math:`R^2` is defined as
        :math:`(1 - \\frac{u}{v})`, where :math:`u` is the residual
        sum of squares ``((y_true - y_pred)** 2).sum()`` and :math:`v`
        is the total sum of squares ``((y_true - y_true.mean()) ** 2).sum()``.
        The best possible score is 1.0 and it can be negative (because the
        model can be arbitrarily worse). A constant model that always predicts
        the expected value of `y`, disregarding the input features, would get
        a :math:`R^2` score of 0.0.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples. For some estimators this may be a precomputed
            kernel matrix or a list of generic objects instead with shape
            ``(n_samples, n_samples_fitted)``, where ``n_samples_fitted``
            is the number of samples used in the fitting for the estimator.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            True values for `X`.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        score : float
            :math:`R^2` of ``self.predict(X)`` w.r.t. `y`.

        Notes
        -----
        The :math:`R^2` score used when calling ``score`` on a regressor uses
        ``multioutput='uniform_average'`` from version 0.23 to keep consistent
        with default value of :func:`~sklearn.metrics.r2_score`.
        This influences the ``score`` method of all the multioutput
        regressors (except for
        :class:`~sklearn.multioutput.MultiOutputRegressor`).
        """

        from sklearn.metrics import r2_score

        y_pred = self.predict(X)
        return r2_score(y, y_pred, sample_weight=sample_weight)


class BalancedEnsembleRandomForestRegressorModel(RegressionModel):
    """
    Ensemble of RandomForestRegressor models predicting a confidence score (see ``MED3pa.med3pa.confidence``), with a
    correction of the imbalance between the two true classes of the base model, both in number of samples and in level
    of errors, so that the errors of both classes have the same importance.

    Each submodel is trained on its own class-balanced undersample of the training data (all the minority class samples
    and a different random subset of the majority class) and its own seed, with sample weights computed on that
    undersample by ``compute_sample_weight``:

    - ``class_balance``: each true class gets the same total weight (``n / (2 * n_k)``).
    - ``error_balance``: the weights of each group ``g`` are multiplied by ``mean_error / mean_error_g``, with
      ``mean_error_g`` the (weighted) mean error ``1 - confidence`` in the group, which removes the difference of error
      level between the groups: the total weighted error of each group becomes proportional to its total weight before
      this correction.
      With ``'true_class'`` (default), the groups are the true classes: the mean errors are the continuous analogues
      of ``1 - sensitivity`` and ``1 - specificity``, which do not depend on the prevalence. With
      ``'predicted_class'``, the groups are the classes predicted by the base model: the mean errors are the continuous
      analogues of ``1 - PPV`` and ``1 - NPV``, which depend on the prevalence and on the base model threshold.
      Combined with ``class_balance``, ``'true_class'`` gives the same total weight and the same total error to both
      true classes.

    If a ``param_grid`` (search space) is given to ``train``, the hyperparameters of the random forests are first
    optimized with ``n_trials`` trials of optuna's TPE sampler, each minimizing the cross-validated (folds stratified
    by true class) weighted MSE on the validation folds with the same weighting, then the ensemble is refitted on all
    the data with the best hyperparameters. In the search space, a list of values is a categorical choice (so a
    GridSearchCV grid can be reused), a tuple ``(low, high)`` an int or float range, ``(low, high, 'log')`` a
    log-scaled range, and an ``optuna.distributions`` distribution is used as is.

    ``predict_uncertainty`` decomposes the predictive variance of the confidence score (law of total variance):

    - aleatoric: mean over the submodels of an out-of-bag estimate of ``Var(confidence | x)``, the noise that more
      data would not remove. For each tree, the out-of-bag samples (not drawn in its bootstrap sample, or left out
      of the undersample of its submodel) falling in the leaf of x give squared residuals to the leaf value,
      corrected for the variance of the leaf value (so the estimate is unbiased whatever the leaf size) and pooled
      over the trees, with the class and error balancing weights of the whole training set. Without out-of-bag
      samples (bootstrap=False and no undersampling), the in-bag variance of the leaves with the small-sample
      correction is used instead, and min_samples_leaf is set to at least 2 so that it is defined.
    - epistemic: variance over the submodels of their mean predictions, the disagreement due to the training data
      each submodel saw and to its randomness.

    ``predict`` returns the confidence under the balanced weighting: errors on both classes count equally, the
    prediction is a cost-weighted confidence (as if the classes were balanced, e.g. it flags the missed positives of an
    imbalanced task). ``predict_debiased`` returns the actual expected confidence E[c | x] instead, e.g. for the shift
    tests, defined on the actual loss. Since the weights only depend on the true class (with
    ``error_balance`` None or 'true_class', and no external sample weights), each submodel learns
    m(x) = (W1 pi c1 + W0 (1 - pi) c0) / (W1 pi + W0 (1 - pi)), with W_k the effective weight of the true class k
    (sampling rate of its undersample times its class and error balancing weight, ``class_weight_ratios_`` = W1 / W0),
    pi(x) = P(y = 1 | x), and c_k(x) the confidence of the base model prediction if y = k (known from its predicted
    probability). Inverting this prior shift gives pi(x), then E[c | x] = pi c1 + (1 - pi) c0.

    The ensemble settings (``n_models``, ``class_balance``, ``error_balance``, ``undersample``, ``sampling_ratio``,
    ``n_trials``) can also be given in ``params`` (e.g. in the ``ipc_params`` of a ``Med3paExperiment``), the other
    parameters are the ones of the RandomForestRegressor submodels.
    """
    ensemble_settings = ('n_models', 'class_balance', 'error_balance', 'undersample', 'sampling_ratio', 'n_trials')
    error_balance_options = (None, 'true_class', 'predicted_class')

    def __init__(self, params: Dict[str, Any] = None,
                 n_models: int = 10,
                 class_balance: bool = True,
                 error_balance: Optional[str] = 'true_class',
                 undersample: bool = True,
                 sampling_ratio: float = 1.0,
                 n_trials: int = 50,
                 random_state: int = None,
                 **params_sklearn) -> None:
        """
        Initializes the BalancedEnsembleRandomForestRegressorModel.

        Args:
            params (Dict[str, Any]): Parameters of the RandomForestRegressor submodels, and possibly ensemble settings
                (which take precedence over the arguments below).
            n_models (int): The number of RandomForestRegressor submodels in the ensemble.
            class_balance (bool): Whether to give the same total weight to both true classes.
            error_balance (Optional[str]): Groups whose mean error levels are equalized: 'true_class', 'predicted_class'
                or None (no error level correction).
            undersample (bool): Whether to train each submodel on a class-balanced undersample of the training data.
            sampling_ratio (float): Number of majority class samples per minority class sample in each undersample.
            n_trials (int): Number of optuna trials of the hyperparameter optimization when a ``param_grid`` is
                given to ``train`` (at most the number of combinations of a categorical-only search space).
            random_state (int): Seed of the undersamples, the submodels and the cross-validation, defaults to the
                'random_state' of params.
            **params_sklearn (Any): Parameters of the RandomForestRegressor submodels.
        """
        super().__init__()
        params = deepcopy(params) if params is not None else {}
        params.update(params_sklearn)
        settings = dict(n_models=n_models, class_balance=class_balance, error_balance=error_balance,
                        undersample=undersample, sampling_ratio=sampling_ratio, n_trials=n_trials)
        for key in self.ensemble_settings:
            settings[key] = params.pop(key, settings[key])
        if settings['error_balance'] not in self.error_balance_options:
            raise ValueError(f"error_balance must be in {self.error_balance_options}, got {settings['error_balance']}")
        if random_state is None:
            random_state = params.get('random_state')
        params.pop('random_state', None)  # each submodel gets its own seed

        self.n_models = settings['n_models']
        self.class_balance = settings['class_balance']
        self.error_balance = settings['error_balance']
        self.undersample = settings['undersample']
        self.sampling_ratio = settings['sampling_ratio']
        self.n_trials = settings['n_trials']
        self.random_state = random_state
        self.params = params
        self.models: List[RandomForestRegressor] = []
        self._leaf_statistics_: List[List[Dict[str, np.ndarray]]] = []
        self.class_weight_ratios_: List[float] = []
        self.best_params_ = None
        self.cv_results_ = None
        self.model = self
        self.model_class = RandomForestRegressor
        self.data_preparation_strategy = ToNumpyStrategy()

        self.fit = self.train

    def __sklearn_clone__(self):
        """
        Overwrites the sklearn clone function
        """
        return deepcopy(self)

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        Returns the parameters of the submodels with the ensemble settings.
        """
        return {**self.params, 'random_state': self.random_state,
                **{key: getattr(self, key) for key in self.ensemble_settings}}

    def compute_sample_weight(self, confidence: np.ndarray, y_true: np.ndarray,
                              y_pred: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Computes the sample weights correcting the class proportions and the error levels (see the class docstring),
        normalized to a mean of 1.

        Args:
            confidence (np.ndarray): Confidence scores of the base model predictions (the regression targets).
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray, optional): Labels predicted by the base model, required with
                error_balance='predicted_class'.

        Returns:
            np.ndarray: The sample weights.
        """
        y_true = np.asarray(y_true).astype(int)
        n = len(y_true)
        weight = np.ones(n)
        if self.class_balance:
            classes, counts = np.unique(y_true, return_counts=True)
            for k, n_k in zip(classes, counts):
                weight[y_true == k] *= n / (len(classes) * n_k)

        if self.error_balance is not None:
            if self.error_balance == 'predicted_class':
                if y_pred is None:
                    raise ValueError("y_pred is required with error_balance='predicted_class'")
                groups = np.asarray(y_pred).astype(int)
            else:
                groups = y_true
            error = np.clip(1.0 - np.asarray(confidence, dtype=float), 0.0, None)
            group_values = np.unique(groups)
            mean_errors = np.array([np.average(error[groups == g], weights=weight[groups == g])
                                    for g in group_values])
            mean_errors = np.maximum(mean_errors, 1e-12)
            for g, mean_error_g in zip(group_values, mean_errors):
                weight[groups == g] *= mean_errors.mean() / mean_error_g

        return weight / weight.mean()

    def _undersample_indices(self, y_true: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Indices of a class-balanced undersample: all the samples of the minority class and a random subset of
        ``sampling_ratio`` times as many samples of the other class.
        """
        classes, counts = np.unique(y_true, return_counts=True)
        if not self.undersample or len(classes) < 2:
            return np.arange(len(y_true))
        n_keep = int(np.ceil(self.sampling_ratio * counts.min()))
        indices = []
        for k, n_k in zip(classes, counts):
            idx_k = np.flatnonzero(y_true == k)
            indices.append(idx_k if n_k <= n_keep else rng.choice(idx_k, size=n_keep, replace=False))
        return np.sort(np.concatenate(indices))

    @staticmethod
    def _in_bag_counts(forest: RandomForestRegressor, tree, n_samples: int) -> np.ndarray:
        """
        Number of times each sample of the training set of the forest was drawn in the bootstrap sample of the tree
        (as drawn in sklearn.ensemble._forest._parallel_build_trees).
        """
        if not forest.bootstrap:
            return np.ones(n_samples)
        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, forest.max_samples)
        indices = _generate_sample_indices(tree.random_state, n_samples, n_samples_bootstrap)
        return np.bincount(indices, minlength=n_samples).astype(float)

    @staticmethod
    def _leaf_statistics(tree, leaves: np.ndarray, y: np.ndarray, in_bag_weight: np.ndarray,
                         oob: np.ndarray, oob_weight: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Statistics of each node of a fitted tree to estimate the aleatoric variance of its leaves.

        With k = sum(w^2) / sum(w)^2 of the in-bag weights w of the leaf (1 / n for n unweighted samples), the
        leaf value (the weighted mean of its in-bag targets) has a variance of k * sigma^2, so:

        - the squared residual to the leaf value of an out-of-bag sample, divided by (1 + k), is an unbiased
          estimate of the variance sigma^2 of the targets in the leaf: 'oob_sum_w' and 'oob_sum_w_residual' are the
          weighted sums of the out-of-bag samples of each leaf and of their corrected squared residuals.
        - the weighted variance of the in-bag targets, divided by (1 - k), also is (the weighted analogue of the
          n / (n - 1) correction), undefined for single-sample leaves (nan): 'in_bag_variance', used when the
          query point falls in leaves without out-of-bag samples in all the trees.

        Args:
            tree: A fitted tree of the forest.
            leaves (np.ndarray): Leaf of each training sample of the submodel in the tree.
            y (np.ndarray): Targets of the training samples.
            in_bag_weight (np.ndarray): Training weight (sample weight times bootstrap count) of each sample in the
                tree, 0 for out-of-bag samples.
            oob (np.ndarray): Boolean mask of the out-of-bag samples of the tree.
            oob_weight (np.ndarray): Weights of the samples in the pooled out-of-bag estimate.
        """
        n_nodes = tree.tree_.node_count
        leaf_value = tree.tree_.value[:, 0, 0]
        sum_w = np.bincount(leaves, weights=in_bag_weight, minlength=n_nodes)
        sum_w2 = np.bincount(leaves, weights=in_bag_weight ** 2, minlength=n_nodes)
        sum_w_residual = np.bincount(leaves, weights=in_bag_weight * (y - leaf_value[leaves]) ** 2, minlength=n_nodes)
        with np.errstate(invalid='ignore', divide='ignore'):
            k = sum_w2 / sum_w ** 2
            in_bag_variance = np.where(k < 1 - 1e-12, sum_w_residual / sum_w / (1 - k), np.nan)
        oob_leaves = leaves[oob]
        oob_residual = (y[oob] - leaf_value[oob_leaves]) ** 2 / (1 + np.nan_to_num(k[oob_leaves]))
        return {'oob_sum_w': np.bincount(oob_leaves, weights=oob_weight[oob], minlength=n_nodes),
                'oob_sum_w_residual': np.bincount(oob_leaves, weights=oob_weight[oob] * oob_residual,
                                                  minlength=n_nodes),
                'in_bag_variance': in_bag_variance}

    def _fit_ensemble(self, x: np.ndarray, y: np.ndarray, y_true: np.ndarray, y_pred: Optional[np.ndarray],
                      sample_weight: Optional[np.ndarray], compute_uncertainty: bool = True) -> None:
        """
        Fits the submodels, each on its own undersample with its own seed, and computes the leaf statistics of the
        aleatoric uncertainty if compute_uncertainty.
        """
        rng = np.random.default_rng(self.random_state)
        seeds = rng.integers(np.iinfo(np.int32).max, size=self.n_models)
        self.models, self._leaf_statistics_, self.class_weight_ratios_ = [], [], []
        # weights of the out-of-bag estimates, with the class and error balancing of the whole training set
        oob_weight = self.compute_sample_weight(y, y_true, y_pred)
        if sample_weight is not None:
            oob_weight = oob_weight * sample_weight
        undersamples = [self._undersample_indices(y_true, np.random.default_rng(seed)) for seed in seeds]
        params = self.params
        if (compute_uncertainty and not params.get('bootstrap', True) and params.get('min_samples_leaf', 1) < 2
                and any(len(idx) == len(y) for idx in undersamples)):
            # no out-of-bag sample: the in-bag variance is undefined for single-sample leaves
            warnings.warn("BalancedEnsembleRandomForestRegressorModel: without bootstrap nor undersampling, "
                          "min_samples_leaf is set to 2 to estimate the aleatoric uncertainty.")
            params = {**params, 'min_samples_leaf': 2}
        for seed, idx in zip(seeds, undersamples):
            weight = self.compute_sample_weight(y[idx], y_true[idx], y_pred[idx] if y_pred is not None else None)
            if sample_weight is not None:
                weight = weight * sample_weight[idx]
            forest = RandomForestRegressor(**params, random_state=int(seed))
            forest.fit(x[idx], y[idx], sample_weight=weight)
            self.models.append(forest)
            self.class_weight_ratios_.append(self._class_weight_ratio(y_true, idx, weight))
            if not compute_uncertainty:
                continue

            forest_statistics = []
            for tree in forest.estimators_:
                in_bag_weight = np.zeros(len(y))
                in_bag_weight[idx] = weight * self._in_bag_counts(forest, tree, len(idx))
                forest_statistics.append(self._leaf_statistics(tree, tree.apply(x), y, in_bag_weight,
                                                               in_bag_weight == 0, oob_weight))
            self._leaf_statistics_.append(forest_statistics)

    def _cv_score(self, params: Dict[str, Any], x: np.ndarray, y: np.ndarray, y_true: np.ndarray,
                  y_pred: Optional[np.ndarray], sample_weight: Optional[np.ndarray],
                  train_idx: np.ndarray, val_idx: np.ndarray) -> float:
        """
        Weighted MSE on a validation fold of the ensemble fitted on the train fold with the given parameters.
        """
        candidate = deepcopy(self)
        candidate.params.update(params)
        sub = (lambda a, idx: a[idx] if a is not None else None)
        candidate._fit_ensemble(x[train_idx], y[train_idx], y_true[train_idx], sub(y_pred, train_idx),
                                sub(sample_weight, train_idx), compute_uncertainty=False)
        weight = self.compute_sample_weight(y[val_idx], y_true[val_idx], sub(y_pred, val_idx))
        if sample_weight is not None:
            weight = weight * sample_weight[val_idx]
        return float(np.average((candidate.predict(x[val_idx]) - y[val_idx]) ** 2, weights=weight))

    @staticmethod
    def _search_space(param_grid: Dict[str, Any]) -> Dict[str, optuna.distributions.BaseDistribution]:
        """
        Optuna distributions of the search space: a list of values is a categorical choice, a tuple ``(low, high)``
        an int (if both bounds are ints) or float range, ``(low, high, 'log')`` a log-scaled range, and an optuna
        distribution is used as is.
        """
        space = {}
        for name, values in param_grid.items():
            if isinstance(values, optuna.distributions.BaseDistribution):
                space[name] = values
            elif isinstance(values, tuple):
                low, high, *scale = values
                log = scale == ['log']
                if isinstance(low, int) and isinstance(high, int):
                    space[name] = optuna.distributions.IntDistribution(low, high, log=log)
                else:
                    space[name] = optuna.distributions.FloatDistribution(float(low), float(high), log=log)
            else:
                space[name] = optuna.distributions.CategoricalDistribution(list(values))
        return space

    def _optimize(self, x: np.ndarray, y: np.ndarray, y_true: np.ndarray, y_pred: Optional[np.ndarray],
                  sample_weight: Optional[np.ndarray], param_grid: Dict[str, Any], cv: int,
                  n_jobs: Optional[int]) -> None:
        """
        Selects the submodels parameters of the param_grid search space minimizing the cross-validated weighted MSE,
        with n_trials trials of an optuna TPE sampler.
        """
        self.models, self._leaf_statistics_ = [], []  # not copied to each candidate
        space = self._search_space(param_grid)
        n_trials = self.n_trials
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=self.random_state))
        if all(isinstance(d, optuna.distributions.CategoricalDistribution) for d in space.values()):
            combinations = list(itertools.product(*[d.choices for d in space.values()]))
            if len(combinations) <= n_trials:  # small search space: try every combination
                n_trials = len(combinations)
                for values in combinations:
                    study.enqueue_trial(dict(zip(space, values)))

        if np.unique(y_true, return_counts=True)[1].min() >= cv:
            splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        folds = list(splitter.split(x, y_true))

        verbosity = optuna.logging.get_verbosity()
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        scores_by_params = {}
        try:
            # the sampler can propose already evaluated categorical values: they are told again without being
            # refitted nor counted in n_trials (bounded number of proposals in case the sampler keeps repeating)
            for _ in range(10 * n_trials):
                if len(scores_by_params) >= n_trials:
                    break
                trial = study.ask(space)
                key = tuple(sorted(trial.params.items(), key=lambda item: item[0]))
                if key not in scores_by_params:
                    scores_by_params[key] = np.array(Parallel(n_jobs=n_jobs)(
                        delayed(self._cv_score)(trial.params, x, y, y_true, y_pred, sample_weight, train_idx,
                                                val_idx)
                        for train_idx, val_idx in folds))
                study.tell(trial, scores_by_params[key].mean())
        finally:
            optuna.logging.set_verbosity(verbosity)

        self.cv_results_ = [{'params': dict(key), 'mean_weighted_mse': s.mean(), 'std_weighted_mse': s.std()}
                            for key, s in scores_by_params.items()]
        self.best_params_ = study.best_params
        self.params.update(self.best_params_)

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None,
              y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None,
              y_true: np.ndarray = None, y_pred: np.ndarray = None, sample_weight: np.ndarray = None,
              param_grid: Optional[Dict[str, Any]] = None, cv: int = 4, n_jobs: Optional[int] = -1) -> None:
        """
        Trains the ensemble, after optimizing its hyperparameters if a param_grid is given.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Confidence scores of the base model predictions for training.
            x_validation (np.ndarray, optional): observations for validation.
            y_validation (np.ndarray, optional): Confidence scores for validation.
            training_parameters (dict, optional): Additional parameters of the RandomForestRegressor submodels.
            y_true (np.ndarray): True labels of the training observations, used to balance the classes.
            y_pred (np.ndarray, optional): Labels predicted by the base model, required with
                error_balance='predicted_class'.
            sample_weight (np.ndarray, optional): Additional sample weights, multiplied with the computed ones.
            param_grid (dict, optional): Search space of the RandomForestRegressor parameters to optimize with optuna
                (see the class docstring).
            cv (int): Number of cross-validation folds of the optimization.
            n_jobs (Optional[int]): Number of jobs running the cross-validation folds of each trial in parallel.
        """
        if y_true is None:
            raise ValueError("BalancedEnsembleRandomForestRegressorModel must be trained with the true labels y_true")
        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)
        np_y_train = np.asarray(np_y_train, dtype=float)
        y_true = np.asarray(y_true).astype(int)
        y_pred = np.asarray(y_pred).astype(int) if y_pred is not None else None
        sample_weight = np.asarray(sample_weight, dtype=float) if sample_weight is not None else None

        if training_parameters:
            valid_param_sets = [set(RandomForestRegressor().get_params())]
            self.params.update(self.validate_params(training_parameters, valid_param_sets))

        if param_grid:
            self._optimize(np_X_train, np_y_train, y_true, y_pred, sample_weight, param_grid, cv, n_jobs)

        self._fit_ensemble(np_X_train, np_y_train, y_true, y_pred, sample_weight)

        if x_validation is not None and y_validation is not None:
            self.evaluate(x_validation, y_validation, ['RMSE', 'MSE'], True)

    @staticmethod
    def _class_weight_ratio(y_true: np.ndarray, idx: np.ndarray, weight: np.ndarray) -> float:
        """
        Effective weight ratio W1 / W0 of the true classes in a submodel trained on the samples idx with the weights
        weight: W_k = (sampling rate of the class k in idx) * (weight of its samples). nan when a class is missing or
        when its weights are not constant (error_balance='predicted_class' or external sample weights), since the
        prior shift is then not a constant per class.
        """
        W = {}
        for k in (0, 1):
            in_k = y_true[idx] == k
            n_k = np.sum(y_true == k)
            if n_k == 0 or not in_k.any() or not np.allclose(weight[in_k], weight[in_k][0]):
                return float('nan')
            W[k] = in_k.sum() / n_k * weight[in_k][0]
        return float(W[1] / W[0])

    def debiasing_available(self) -> bool:
        """Whether ``predict_debiased`` can be used (the prior shift of every submodel is a constant per class)."""
        ratios = getattr(self, 'class_weight_ratios_', [])
        return bool(self.models) and len(ratios) == len(self.models) and bool(np.all(np.isfinite(ratios)))

    def predict_debiased(self, X: np.ndarray, c0: np.ndarray, c1: np.ndarray,
                         return_risk: bool = False) -> np.ndarray | Tuple[np.ndarray, np.ndarray]:
        """
        Predicts the actual expected confidence E[c | x], removing the prior shift of the class and error balancing
        (see the class docstring), from the confidence the base model predictions would have if y = 0 and if y = 1.

        Args:
            X (np.ndarray): observations for prediction.
            c0 (np.ndarray): confidence of the base model prediction of each observation if its label is 0.
            c1 (np.ndarray): confidence of the base model prediction of each observation if its label is 1.
            return_risk (bool): also return the estimated risk pi(x) = P(y = 1 | x).

        Returns:
            np.ndarray: E[c | x] (mean over the submodels), and pi(x) if return_risk.
        """
        self._check_fitted()
        if not self.debiasing_available():
            raise ValueError("predict_debiased requires weights constant within each true class in every submodel: "
                             "error_balance None or 'true_class', no external sample_weight, both classes present.")
        np_X, _ = self._ensure_numpy_arrays(X)
        c0, c1 = np.asarray(c0, dtype=float), np.asarray(c1, dtype=float)
        low, high = np.minimum(c0, c1), np.maximum(c0, c1)
        same = np.abs(c0 - c1) < 1e-9  # any pi gives the same confidence
        confidences, risks = [], []
        for forest, ratio in zip(self.models, self.class_weight_ratios_):
            m = np.clip(forest.predict(np_X), low, high)
            with np.errstate(divide='ignore', invalid='ignore'):
                odds = (c0 - m) / (m - c1) / ratio  # pi / (1 - pi), from W1 pi / (W0 (1 - pi))
                risk = np.where(np.isfinite(odds), odds / (1.0 + odds), 1.0)
            risk = np.where(same, 0.5, risk)
            confidences.append(risk * c1 + (1.0 - risk) * c0)
            risks.append(risk)
        confidence = np.mean(confidences, axis=0)
        return (confidence, np.mean(risks, axis=0)) if return_risk else confidence

    def _check_fitted(self) -> None:
        if not self.models:
            raise ValueError("The BalancedEnsembleRandomForestRegressorModel has not been trained.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the confidence scores as the mean of the submodels predictions.
        """
        self._check_fitted()
        np_X, _ = self._ensure_numpy_arrays(X)
        return np.mean([forest.predict(np_X) for forest in self.models], axis=0)

    @staticmethod
    def _forest_aleatoric(forest: RandomForestRegressor, statistics: List[Dict[str, np.ndarray]],
                          X: np.ndarray) -> np.ndarray:
        """
        Aleatoric variance of a submodel at X: the corrected squared residuals of the out-of-bag samples in the
        leaves of X, pooled over the trees, else the mean over the trees of the corrected in-bag variance of the
        leaves of X (see ``_leaf_statistics``).
        """
        leaves = forest.apply(X)  # (n_samples, n_trees)
        sum_w = np.zeros(len(X))
        sum_w_residual = np.zeros(len(X))
        in_bag_variance = np.empty((len(X), len(statistics)))
        for t, tree_statistics in enumerate(statistics):
            sum_w += tree_statistics['oob_sum_w'][leaves[:, t]]
            sum_w_residual += tree_statistics['oob_sum_w_residual'][leaves[:, t]]
            in_bag_variance[:, t] = tree_statistics['in_bag_variance'][leaves[:, t]]
        with np.errstate(invalid='ignore', divide='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)  # mean of all-nan rows
            fallback = np.nan_to_num(np.nanmean(in_bag_variance, axis=1))
            return np.where(sum_w > 0, sum_w_residual / sum_w, fallback)

    def predict_uncertainty(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Predicts the confidence scores with their aleatoric and epistemic uncertainty (see the class docstring).

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            Dict[str, np.ndarray]: 'mean' (the predictions), 'aleatoric' and 'epistemic' variances, and 'total'
                variance (their sum).
        """
        self._check_fitted()
        np_X, _ = self._ensure_numpy_arrays(X)
        means = np.array([forest.predict(np_X) for forest in self.models])
        aleatoric = np.mean([self._forest_aleatoric(forest, statistics, np_X)
                             for forest, statistics in zip(self.models, self._leaf_statistics_)], axis=0)
        epistemic = means.var(axis=0, ddof=1) if len(self.models) > 1 else np.zeros(len(np_X))
        return {'mean': means.mean(axis=0), 'aleatoric': aleatoric, 'epistemic': epistemic,
                'total': aleatoric + epistemic}

    def score(self, X, y, sample_weight=None):
        """
        Return the coefficient of determination R^2 of the prediction (see EnsembleRandomForestRegressorModel.score).
        """
        from sklearn.metrics import r2_score

        return r2_score(y, self.predict(X), sample_weight=sample_weight)


class DecisionTreeRegressorModel(RegressionModel):
    """
    A concrete implementation of the Model class for DecisionTree models.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the DecisionTreeRegressorModel with a scikit-learn DecisionTreeRegressor.

        Args:
            params (dict): Parameters for initializing the DecisionTreeRegressor.
        """
        super().__init__()
        self.params = params
        self.model = DecisionTreeRegressor(**params)
        self.model_class = DecisionTreeRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None, y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Targets for training.
            x_validation (np.ndarray, optional): observations for validation.
            y_validation (np.ndarray, optional): Targets for validation.
            training_parameters (dict, optional): Additional training parameters.

        Raises:
            ValueError: If the DecisionTreeRegressorModel has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The DecisionTreeRegressorModel has not been initialized.")

        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)

        if training_parameters:
            valid_param_sets = [set(self.model.get_params())]
            validated_params = self.validate_params(training_parameters, valid_param_sets)
            self.params.update(validated_params)
            self.model.set_params(**self.params)
        
        self.model.fit(np_X_train, np_y_train)

        if x_validation is not None and y_validation is not None:
            self.evaluate(x_validation, y_validation, ['RMSE', 'MSE'], True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the model for the given input.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: Predictions made by the model.

        Raises:
            ValueError: If the DecisionTreeRegressorModel has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The DecisionTreeRegressorModel has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)
