"""
Similar to ``concrete_classifiers.py``, this module contains implementations of regression models like RandomForestRegressor and DecisionTreeRegressor. 
It provides practical, ready-to-use models that comply with the abstract definitions, making it easier to integrate and use these models in ``med3pa`` and ``detectron``.
"""
from typing import Any, Dict, List, Optional

import numpy as np
# from sklearn.base import clone
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample

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

    # def _ensure_numpy_arrays(self, observations: Any, labels: Optional[np.ndarray] = None) -> tuple:
    #     """
    #     Ensures that the input data is converted to NumPy array format, using the defined data preparation strategy.
    #     This method is used internally to standardize input data before training, predicting, or evaluating.
    #
    #     Args:
    #         observations (Any): observations data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
    #         labels (np.ndarray, optional): Labels data, similar to observations in that it can be in various formats. If labels are not provided,
    #                                     only observations are converted and returned.
    #
    #     Returns:
    #         tuple: The observations and labels (if provided) as NumPy arrays. If labels are not provided, labels in the tuple will be None.
    #     """
    #     if not isinstance(observations, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
    #         return self.data_preparation_strategy.execute(observations, labels)
    #     else:
    #         return observations, labels

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
            valid_param_sets = [set(self.model.get_params().keys())]
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
                 random_state: int = None, **params_sklearn) -> None:
        """
        Initializes the EnsembleRandomForestRegressorModel with multiple RandomForestRegressor models.

        Args:
            base_model (RandomForestRegressorModel): A prototype instance of RandomForestRegressorModel.
            n_models (int): The number of RandomForestRegressorModel instances in the ensemble.
            params (Dict[str, Any]): A list of parameter dictionaries for each model in the ensemble.
            random_state (int): A random_state can be set for reproducibility
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
        # for params in params:
        #     model = deepcopy(base_model)(params)
        #     # model.params = params
        #     # model.model = model.model_class(**params)  # reinitialize the model with new params
        #     self.models.append(model)
        self.model = self

        self.fit = self.train

    # def _ensure_numpy_arrays(self, observations: Any, labels: Optional[np.ndarray] = None) -> tuple:
    #     """
    #     Ensures that the input data is converted to NumPy array format.
    #     """
    #     if not isinstance(observations, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
    #         observations, labels = self.models[0].data_preparation_strategy.execute(observations, labels)
    #     return observations, labels

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

    # def _undersample(self, x: np.ndarray, y: np.ndarray) -> tuple:
    #     """
    #     Applies undersampling to the majority class.
    #     """
    #     # Identify the majority and minority classes
    #     unique_classes, class_counts = np.unique(y, return_counts=True)
    #     minority_class = unique_classes[np.argmin(class_counts)]
    #     majority_class = unique_classes[np.argmax(class_counts)]
    #
    #     # Separate the data by class
    #     x_minority = x[y == minority_class]
    #     y_minority = y[y == minority_class]
    #     x_majority = x[y == majority_class]
    #     y_majority = y[y == majority_class]
    #
    #     # Undersample the majority class
    #     x_majority_resampled, y_majority_resampled = resample(x_majority, y_majority,
    #                                                           replace=False,
    #                                                           n_samples=len(y_minority),
    #                                                           random_state=self.random_state)
    #
    #     # Combine minority and resampled majority class
    #     x_resampled = np.vstack((x_minority, x_majority_resampled))
    #     y_resampled = np.hstack((y_minority, y_majority_resampled))
    #
    #     return x_resampled, y_resampled

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

        # if params is not None:
        #     if training_parameters is None:
        #         training_parameters = params
        #     else:
        #         training_parameters.update(params)

        if training_parameters:
            self.params.update(training_parameters)
        if "sample_weight" not in params:
            raise ValueError("EnsembleRandomForestRegressorModel must be trained with a sample_weight parameter")
        sample_weight = params["sample_weight"]

        for model in self.models:
            # Resample the dataset for each model
            x_resampled, y_resampled, sample_weight = self._undersample(np_X_train, np_y_train, sample_weight)
            model.train(x_resampled, y_resampled, x_validation, y_validation, training_parameters, **params)

    # def fit(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None, y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None) -> None:
    #     self.train(x_train, y_train, x_validation, y_validation, training_parameters)

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

    # def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
    #     """
    #     Evaluates the ensemble model using specified metrics by aggregating predictions.
    #     """
    #     predictions = self.predict(X)
    #     evaluation_results = {}
    #
    #     for metric_name in eval_metrics:
    #         metric_function = RegressionEvaluationMetrics.get_metric(metric_name)
    #         if metric_function:
    #             evaluation_results[metric_name] = metric_function(y, predictions)
    #         else:
    #             print(f"Error: The metric '{metric_name}' is not supported.")
    #
    #     if print_results:
    #         self.print_evaluation_results(results=evaluation_results)
    #
    #     return evaluation_results


# class EnsembleUndersamplingRandomForestRegressorModel(RegressionModel):
#     """
#     A concrete implementation of the Model class for Ensemble RandomForestRegressor models with undersampling.
#     """
#
#     def __init__(self, params: Dict[str, Any]) -> None:
#         """
#         Initializes the RandomForestRegressorModel with a scikit-learn RandomForestRegressor.
#
#         Args:
#             params (dict): Parameters for initializing the RandomForestRegressor.
#         """
#         super().__init__()
#         self.params = params
#         self.model = RandomForestRegressor(**params)
#         self.model_class = RandomForestRegressor
#         self.pickled_model = False
#
#     def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray = None,
#               y_validation: np.ndarray = None, training_parameters: Optional[Dict[str, Any]] = None) -> None:
#         """
#         Trains the model on the provided dataset.
#
#         Args:
#             x_train (np.ndarray): observations for training.
#             y_train (np.ndarray): Labels for training.
#             x_validation (np.ndarray, optional): observations for validation.
#             y_validation (np.ndarray, optional): Labels for validation.
#             training_parameters (dict, optional): Additional training parameters.
#
#         Raises:
#             ValueError: If the RandomForestRegressorModel has not been initialized before training.
#         """
#         if self.model is None:
#             raise ValueError("The RandomForestRegressor has not been initialized.")
#
#         np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)
#
#         if training_parameters:
#             valid_param_sets = [set(self.model.get_params().keys())]
#             validated_params = self.validate_params(training_parameters, valid_param_sets)
#             self.params.update(validated_params)
#             self.model.set_params(**self.params)
#
#         self.model.fit(np_X_train, np_y_train)
#
#         if x_validation is not None and y_validation is not None:
#             self.evaluate(x_validation, y_validation, ['RMSE', 'MSE'], True)
#
#     def predict(self, X: np.ndarray) -> np.ndarray:
#         """
#         Makes predictions with the model for the given input.
#
#         Args:
#             X (np.ndarray): observations for prediction.
#
#         Returns:
#             np.ndarray: Predictions made by the model.
#
#         Raises:
#             ValueError: If the RandomForestRegressorModel has not been initialized before training.
#         """
#         if self.model is None:
#             raise ValueError("The RandomForestRegressorModel has not been initialized.")
#         else:
#             np_X, _ = self._ensure_numpy_arrays(X)
#             return self.model.predict(np_X)
#
#     def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[
#         str, float]:
#         """
#         Evaluates the model using specified metrics.
#
#         Args:
#             X (np.ndarray): observations for evaluation.
#             y (np.ndarray): True labels for evaluation.
#             eval_metrics (List[str]): Metrics to use for evaluation.
#             print_results (bool, optional): Whether to print the evaluation results.
#
#         Returns:
#             Dict[str, float]: A dictionary with metric names and their evaluated scores.
#
#         Raises:
#             ValueError: If the model has not been trained before evaluation.
#         """
#         if self.model is None:
#             raise ValueError("Model must be trained before evaluation.")
#
#         predictions = self.predict(X)
#         evaluation_results = {}
#
#         for metric_name in eval_metrics:
#             metric_function = RegressionEvaluationMetrics.get_metric(metric_name)
#             if metric_function:
#                 evaluation_results[metric_name] = metric_function(y, predictions)
#             else:
#                 print(f"Error: The metric '{metric_name}' is not supported.")
#
#         if print_results:
#             self.print_evaluation_results(results=evaluation_results)
#
#         return evaluation_results


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

    # def _ensure_numpy_arrays(self, observations: Any, labels: Optional[np.ndarray] = None) -> tuple:
    #     """
    #     Ensures that the input data is converted to NumPy array format, using the defined data preparation strategy.
    #     This method is used internally to standardize input data before training, predicting, or evaluating.
    #
    #     Args:
    #         observations (Any): observations data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
    #         labels (np.ndarray, optional): Labels data, similar to observations in that it can be in various formats. If labels are not provided,
    #                                     only observations are converted and returned.
    #
    #     Returns:
    #         tuple: The observations and labels (if provided) as NumPy arrays. If labels are not provided, labels in the tuple will be None.
    #     """
    #     if not isinstance(observations, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
    #         return self.data_preparation_strategy.execute(observations, labels)
    #     else:
    #         return observations, labels

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
            valid_param_sets = [set(self.model.get_params().keys())]
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

    # def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
    #     """
    #     Evaluates the model using specified metrics.
    #
    #     Args:
    #         X (np.ndarray): observations for evaluation.
    #         y (np.ndarray): True labels for evaluation.
    #         eval_metrics (List[str]): Metrics to use for evaluation.
    #         print_results (bool, optional): Whether to print the evaluation results.
    #
    #     Returns:
    #         Dict[str, float]: A dictionary with metric names and their evaluated scores.
    #
    #     Raises:
    #         ValueError: If the model has not been trained before evaluation.
    #     """
    #     if self.model is None:
    #         raise ValueError("Model must be trained before evaluation.")
    #
    #     predictions = self.predict(X)
    #     evaluation_results = {}
    #
    #     for metric_name in eval_metrics:
    #         metric_function = RegressionEvaluationMetrics.get_metric(metric_name)
    #         if metric_function:
    #             evaluation_results[metric_name] = metric_function(y, predictions)
    #         else:
    #             print(f"Error: The metric '{metric_name}' is not supported.")
    #
    #     if print_results:
    #         self.print_evaluation_results(results=evaluation_results)
    #
    #     return evaluation_results
    