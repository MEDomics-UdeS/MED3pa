"""
This module defines the abstract structures of models in a machine learning framework. It provides abstract classes for different model types including general, classification, and regression models.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod
from .data_strategies import DataPreparingStrategy

class Model(ABC):
    """
    An abstract base class for all models, defining a common API for model operations such as evaluation and parameter validation.

    Attributes:
        model (Any): The underlying model instance.
        model_class (type): The class type of the underlying model instance.
        params (dict): The params used for initializinf the model.
        data_preparation_strategy (DataPreparingStrategy): Strategy for preparing data before training or evaluation.
    """

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): Features for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool, optional): Whether to print the evaluation results. Defaults to False.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.
        """
        pass

    def print_evaluation_results(self, results: Dict[str, float]) -> None:
        """
        Prints the evaluation results in a formatted manner.

        Args:
            results (Dict[str, float]): A dictionary with metric names and their evaluated scores.
        """
        print("Evaluation Results:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}")

    def validate_params(self, params: Dict[str, Any], valid_param_sets: List[set]) -> Dict[str, Any]:
        """
        Validates the model parameters against a list of valid parameter sets.

        Args:
            params (Dict[str, Any]): Parameters to validate.
            valid_param_sets (List[set]): A list of sets containing valid parameter names.

        Returns:
            Dict[str, Any]: Validated parameters.

        Raises:
            ValueError: If any invalid parameters are found.
        """
        combined_valid_params = set().union(*valid_param_sets)
        invalid_params = [k for k in params.keys() if k not in combined_valid_params]
        if invalid_params:
            raise ValueError(f"Invalid parameters found: {invalid_params}")
        return {k: v for k, v in params.items() if k in combined_valid_params}

    def get_data_strategy(self) -> DataPreparingStrategy:
        """
        Retrieves the data preparation strategy used by the model.

        Returns:
            DataPreparingStrategy: The data preparation strategy.
        """
        return self.data_preparation_strategy

    def set_model(self, model: Any) -> None:
        """
        Sets the underlying model instance.

        Args:
            model (Any): The model instance to be used.
        """
        self.model = model
        self.model_class = type(model)

class ClassificationModel(Model):
    """
    Abstract base class for classification models, extending the generic Model class with additional classification-specific methods.
    """

    def balance_train_weights(self, y_train: np.ndarray) -> np.ndarray:
        """
        Balances the training weights based on the class distribution in the training data.

        Args:
            y_train (np.ndarray): Labels for training.

        Returns:
            np.ndarray: Balanced training weights.

        Raises:
            AssertionError: If balancing is attempted on non-binary classification data.
        """
        _, counts = np.unique(y_train, return_counts=True)
        assert len(counts) == 2, 'Only binary classification is supported'
        c_neg, c_pos = counts[0], counts[1]
        pos_weight, neg_weight = 2 * c_neg / (c_neg + c_pos), 2 * c_pos / (c_neg + c_pos)
        train_weights = np.array([pos_weight if label == 1 else neg_weight for label in y_train])
        return train_weights

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]], balance_train_classes: bool) -> None:
        """
        Trains the classification model using provided training and validation data.

        Args:
            x_train (np.ndarray): Features for training.
            y_train (np.ndarray): Labels for training.
            x_validation (np.ndarray): Features for validation.
            y_validation (np.ndarray): Labels for validation.
            training_parameters (Dict[str, Any], optional): Additional training parameters.
            balance_train_classes (bool): Whether to balance the training classes.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray, return_proba: bool = False, threshold: float = 0.5) -> np.ndarray:
        """
        Makes predictions for the given input features.

        Args:
            X (np.ndarray): Features for prediction.
            return_proba (bool, optional): Whether to return probabilities instead of class labels. Defaults to False.
            threshold (float, optional): Threshold for converting probabilities to class labels. Defaults to 0.5.

        Returns:
            np.ndarray: The predicted labels or probabilities.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

class RegressionModel(Model):
    """
    Abstract base class for regression models, providing a framework for training and prediction in regression tasks.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, x_validation: np.ndarray, y_validation: np.ndarray, training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the regression model using provided training and validation data.

        Args:
            x_train (np.ndarray): Features for training.
            y_train (np.ndarray): Labels for training.
            x_validation (np.ndarray): Features for validation.
            y_validation (np.ndarray): Labels for validation.
            training_parameters (Dict[str, Any], optional): Additional training parameters.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the given input features.

        Args:
            X (np.ndarray): Features for prediction.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass
