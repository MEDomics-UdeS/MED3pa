from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from .data_strategies import ToNumpyStrategy
from .abstract_models import RegressionModel
from .regression_metrics import *
from typing import Optional, Dict, Any, List

class RandomForestRegressorModel(RegressionModel):
    """
    A concrete implementation of the Model class for RandomForestRegressor models.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the RandomForestRegressorModel with a sklearn RandomForestRegressor.

        Parameters
        ----------
        params : dict
            Parameters for initializing the RandomForestRegressor.
        """
        self.params = params
        self.model = RandomForestRegressor(**params)
        self.model_class = RandomForestRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def _ensure_numpy_arrays(self, features: Any, labels: Optional[np.ndarray] = None) -> tuple:
        """
        Ensures that the input data is converted to NumPy array format.

        Parameters
        ----------
        features : Any
            Features data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
        labels : np.ndarray, optional
            Labels data, similar to features in that it can be in various formats.

        Returns
        -------
        tuple
            The features and labels (if provided) as NumPy arrays.
        """
        if not isinstance(features, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
            return self.data_preparation_strategy.execute(features, labels)
        else:
            return features, labels

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the model on the provided dataset.

        Parameters
        ----------
        x_train : np.ndarray
            Features for training.
        y_train : np.ndarray
            Labels for training.
        x_validation : np.ndarray
            Features for validation.
        y_validation : np.ndarray
            Labels for validation.
        training_parameters : dict, optional
            Additional training parameters.

        Raises
        ------
        ValueError
            If the RandomForestRegressor has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The RandomForestRegressor has not been initialized.")

        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)

        if training_parameters:
            valid_params = self.model.get_params()
            for k in training_parameters.keys():
                if k not in valid_params:
                    raise ValueError(f"Invalid parameter: {k}")
            self.params.update(training_parameters)
            self.model.set_params(**self.params)

        self.model.fit(np_X_train, np_y_train)

        if x_validation is not None and y_validation is not None:
            np_X_val, np_y_val = self._ensure_numpy_arrays(x_validation, y_validation)
            val_predictions = self.model.predict(np_X_val)
            val_score = self.model.score(np_X_val, np_y_val)
            print(f"Validation score: {val_score}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the model for the given input.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.

        Returns
        -------
        np.ndarray
            Predictions made by the model.

        Raises
        ------
        ValueError
            If the RandomForestRegressor has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The RandomForestRegressor has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Parameters
        ----------
        X : np.ndarray
            Features for evaluation.
        y : np.ndarray
            True labels for evaluation.
        eval_metrics : list of str
            Metrics to use for evaluation.
        print_results : bool, optional
            Whether to print the evaluation results. Defaults to False.

        Returns
        -------
        dict
            A dictionary with metric names and their evaluated scores.

        Raises
        ------
        ValueError
            If the model has not been trained before evaluation.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.predict(X)
        evaluation_results = {}

        for metric_name in eval_metrics:
            metric_class = regression_metrics_mappings.get(metric_name)
            if metric_class:
                metric = metric_class
                evaluation_results[metric_name] = metric.calculate(y, predictions)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            self.print_evaluation_results(results=evaluation_results)

        return evaluation_results

class DecisionTreeRegressorModel(RegressionModel):
    """
    A concrete implementation of the Model class for DecisionTree models.
    """
    def __init__(self, params: Dict[str, Any]) -> None:
        """
        Initializes the DecisionTreeRegressorModel with a sklearn DecisionTreeRegressor.

        Parameters
        ----------
        params : dict
            Parameters for initializing the DecisionTreeRegressor.
        """
        self.params = params
        self.model = DecisionTreeRegressor(**params)
        self.model_class = DecisionTreeRegressor
        self.pickled_model = False
        self.data_preparation_strategy = ToNumpyStrategy()

    def _ensure_numpy_arrays(self, features: Any, labels: Optional[np.ndarray] = None) -> tuple:
        """
        Ensures that the input data is converted to NumPy array format.

        Parameters
        ----------
        features : Any
            Features data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
        labels : np.ndarray, optional
            Labels data, similar to features in that it can be in various formats.

        Returns
        -------
        tuple
            The features and labels (if provided) as NumPy arrays.
        """
        if not isinstance(features, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
            return self.data_preparation_strategy.execute(features, labels)
        else:
            return features, labels

    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the model on the provided dataset.

        Parameters
        ----------
        x_train : np.ndarray
            Features for training.
        y_train : np.ndarray
            Labels for training.
        x_validation : np.ndarray
            Features for validation.
        y_validation : np.ndarray
            Labels for validation.
        training_parameters : dict, optional
            Additional training parameters.

        Raises
        ------
        ValueError
            If the DecisionTreeRegressor has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The RandomForestRegressor has not been initialized.")

        np_X_train, np_y_train = self._ensure_numpy_arrays(x_train, y_train)

        if training_parameters:
            valid_params = self.model.get_params()
            for k in training_parameters.keys():
                if k not in valid_params:
                    raise ValueError(f"Invalid parameter: {k}")
            self.params.update(training_parameters)
            self.model.set_params(**self.params)

        self.model.fit(np_X_train, np_y_train)

        if x_validation is not None and y_validation is not None:
            np_X_val, np_y_val = self._ensure_numpy_arrays(x_validation, y_validation)
            val_predictions = self.model.predict(np_X_val)
            val_score = self.model.score(np_X_val, np_y_val)
            print(f"Validation score: {val_score}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions with the model for the given input.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.

        Returns
        -------
        np.ndarray
            Predictions made by the model.

        Raises
        ------
        ValueError
            If the DecisionTreeRegressor has not been initialized before training.
        """
        if self.model is None:
            raise ValueError("The DecisionTreeRegressor has not been initialized.")
        else:
            np_X, _ = self._ensure_numpy_arrays(X)
            return self.model.predict(np_X)

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Parameters
        ----------
        X : np.ndarray
            Features for evaluation.
        y : np.ndarray
            True labels for evaluation.
        eval_metrics : list of str
            Metrics to use for evaluation.
        print_results : bool, optional
            Whether to print the evaluation results. Defaults to False.

        Returns
        -------
        dict
            A dictionary with metric names and their evaluated scores.

        Raises
        ------
        ValueError
            If the model has not been trained before evaluation.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.predict(X)
        evaluation_results = {}

        for metric_name in eval_metrics:
            metric_class = regression_metrics_mappings.get(metric_name)
            if metric_class:
                metric = metric_class
                evaluation_results[metric_name] = metric.calculate(y, predictions)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            self.print_evaluation_results(results=evaluation_results)

        return evaluation_results