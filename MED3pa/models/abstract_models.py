"""
The abstract_models.py module defines core abstract classes that serve as the foundation for model management in the system. 
It includes ``Model``, which standardizes basic operations like evaluation and parameter validation..etc across all models. 
It also introduces specialized abstract classes such as ``ClassificationModel`` and ``RegressionModel``, 
each adapting these operations to specific needs of classification and regression tasks.
"""

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict, List, Optional, Union
import os
import pickle
import json

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import roc_curve, precision_recall_curve, balanced_accuracy_score, recall_score, roc_auc_score, \
    average_precision_score, matthews_corrcoef, f1_score, precision_score, RocCurveDisplay

from .data_strategies import DataPreparingStrategy

from .classification_metrics import *
from .regression_metrics import *


class Model(ABC, BaseEstimator):
    """
    An abstract base class for all models, defining a common API for model operations such as evaluation and parameter validation.

    Attributes:
        model (Any): The underlying model instance.
        model_class (type): The class type of the underlying model instance.
        params (dict): The params used for initializing the model.
        data_preparation_strategy (DataPreparingStrategy): Strategy for preparing data before training or evaluation.
        pickled_model (Boolean): A boolean indicating whether the model has been loaded from a pickled file.
    """

    def __init__(self, random_state: int = None) -> None:
        super().__init__()
        self.model = None
        self.model_class = None
        self.params = None
        self.data_preparation_strategy = None
        self.pickled_model = False
        self.file_path = None
        self._random_state = random_state

    def get_model(self) -> Any:
        """
        Retrieves the underlying model instance, which is typically a machine learning model object.

        Returns:
            Any: The underlying model instance if set, None otherwise.
        """
        return self.model

    def get_path(self) -> Any:
        """
        Retrieves the file path of the model if it has been loaded from a pickled file.

        Returns:
            str: The file path of the model if it has been loaded from a pickled file, None otherwise.
        """
        return self.file_path

    def get_model_type(self) -> Optional[str]:
        """
        Retrieves the class type of the underlying model instance, which indicates the specific 
        implementation of the model used.

        Returns:
            Optional[str]: The class of the model if set, None otherwise.
        """
        return self.model_class.__name__ if self.model_class else None

    def get_data_strategy(self) -> Optional[str]:
        """
        Retrieves the data preparation strategy associated with the model. This strategy handles 
        how data should be formatted before being passed to the model for training or evaluation.

        Returns:
            Optional[str]: The name of the current data preparation strategy if set, None otherwise.
        """
        return self.data_preparation_strategy.__class__.__name__ if self.data_preparation_strategy else None

    def get_params(self, deep: bool = False) -> Dict[str, Any]:
        """
        Retrieves the underlying model's parameters.

        Returns:
            Dict[str, Any]: the model's parameters.
        """
        return self.params

    def is_pickled(self):
        """
        Returns whether or not the model has been loaded from a pickled file.
        
        Returns:
            Boolean: has the model been loaded from a pickled file.
        """
        return self.pickled_model

    def set_model(self, model: Any) -> None:
        """
        Sets the underlying model instance and updates the model class to match the type of the given model.

        Args:
            model (Any): The model instance to be set.
        """
        self.model = model
        self.model_class = type(model)

    def set_params(self, params: dict = None, **kwargs) -> None:
        """
        Sets the parameters for the model. These parameters are typically used for model initialization or configuration.

        Args:
            params (Dict[str, Any]): A dictionary of parameters for the model.
        """
        if params is None:
            params = {}
        if kwargs is not None:
            params.update(kwargs)
        self.params = params
        return self

    def set_file_path(self, file_path: str):
        """
        Sets the file path of the model. 

        Args:
            file_path (str): the file path of the model.
        """
        self.file_path = file_path

    def update_params(self, params: dict):
        """
        Updates the current model parameters by merging new parameter values from the given dictionary.
        This method allows for dynamic adjustment of model configuration during runtime.

        Args:
            params (Dict[str, Any]): A dictionary containing parameter names and values to be updated.
        """
        self.params.update(params)

    def set_data_strategy(self, strategy: DataPreparingStrategy):
        """
        Sets the underlying model's data preparation strategy.

        Args:
            strategy (DataPreparingStrategy): strategy to be used to prepare the data for training, validation...etc.
        """
        self.data_preparation_strategy = strategy

    def get_info(self) -> Dict[str, Any]:
        """
        Retrieves detailed information about the model.

        Returns:
            Dict[str, Any]: A dictionary containing information about the model's type, parameters, 
                            data preparation strategy, and whether it's a pickled model.
        """
        return {
            "model": self.__class__.__name__,
            "model_type": self.get_model_type(),
            "params": self.get_params(),
            "data_preparation_strategy": self.get_data_strategy() if self.get_data_strategy() else None,
            "pickled_model": self.is_pickled(),
            "file_path": self.get_path()
        }

    def save(self, path: str) -> None:
        """
        Saves the model instance as a pickled file and the parameters as a JSON file within the specified directory.
        
        Args:
            path (str): The directory path where the model and parameters will be saved.
        """
        # Create the directory if it does not exist
        if not os.path.exists(path):
            os.makedirs(path)

        # Define file paths
        model_path = os.path.join(path, 'model_instance.pkl')
        params_path = os.path.join(path, 'model_info.json')

        # Save the model as a pickled file
        with open(model_path, 'wb') as model_file:
            pickle.dump(self.model, model_file)

        # Save the parameters as a JSON file
        with open(params_path, 'w') as params_file:
            json.dump(self.get_info(), params_file)

    @abstractmethod
    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False
                 ) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): observations for evaluation.
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


class ClassificationModel(Model, ClassifierMixin):
    """
    Abstract base class for classification models, extending the generic Model class with additional
    classification-specific methods.
    """

    def __init__(self, objective: str = 'binary:logistic', class_weighting: bool = False, random_state: int = None):
        super().__init__(random_state=random_state)
        self._objective = objective
        self._class_weighting = class_weighting
        self._threshold = 0.5
        self._calibration = None
        self.classes_ = np.array([0, 1])  # To allow model calibration. The CalibratedClassifierCV uses this variable to

    # ensure that the model to be calibrated has been fitted.

    @property
    def threshold(self):
        return self._threshold

    @staticmethod
    def balance_train_weights(y_train: np.ndarray) -> np.ndarray:
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

    def calibrate_model(self, y_pred, y_true, data=None, method='sklearn'):
        if method == 'sklearn':
            calibration = CalibratedClassifierCV(estimator=deepcopy(self), method='sigmoid', cv='prefit')
            calibration.fit(data, y_true)
        else:
            raise NotImplementedError
        self._calibration = calibration

    @abstractmethod
    def fit(self, X, y, threshold: str = None, calibrate: bool = False, *params):
        """

        """
        pass

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray, training_parameters: Optional[Dict[str, Any]],
              balance_train_classes: bool, weights: np.ndarray = None, *params) -> None:
        """
        Trains the classification model using provided training and validation data.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            training_parameters (Dict[str, Any], optional): Additional training parameters.
            balance_train_classes (bool): Whether to balance the training classes.
            weights (Optional[np.ndarray], optional): Weights for the training data
            *params : Additional training parameters for specific models

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    def __init_subclass__(cls, **kwargs):
        """
        Initializes the fit and train methods so only one needs to be defined if child classes
        """
        super().__init_subclass__(**kwargs)
        # Check if either fit or train has been overridden in the subclass
        if 'fit' not in cls.__dict__ and 'train' not in cls.__dict__:
            raise TypeError(f"Class {cls.__name__} must implement either `fit` or `train`.")

        if 'fit' not in cls.__dict__:
            cls.fit = cls.train
        if 'train' not in cls.__dict__:
            cls.train = cls.fit

    def train_to_disagree(self, x_train: np.ndarray, y_train: np.ndarray,
                          x_test: np.ndarray, y_test: np.ndarray,
                          training_parameters: Optional[Dict[str, Any]], balance_train_classes: bool) -> None:
        """
        Trains the classification model using provided training and validation data.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            x_test (np.ndarray): observations for testing.
            y_test (np.ndarray): Labels for testing.
            training_parameters (Dict[str, Any], optional): Additional training parameters.
            balance_train_classes (bool): Whether to balance the training classes.

        """
        N = len(y_test)

        # if additional training parameters are provided
        if training_parameters:
            training_weights = self.balance_train_weights(y_train) if balance_train_classes else (
                training_parameters.get('training_weights', np.ones_like(y_train)))
        else:
            training_weights = np.ones_like(y_train)

        # prepare the data for training
        data = np.concatenate([x_train, x_test])
        label = np.concatenate([y_train, 1 - y_test])
        weights = np.concatenate([training_weights, 1 / (N + 1) * np.ones(N)])

        self.train(data, label, training_parameters=training_parameters,
                   balance_train_classes=balance_train_classes, weights=weights)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes label predictions for the given input observations.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: The predicted labels or probabilities.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Makes probability predictions for the given input observations.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: The predicted labels or probabilities.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: Union[str, List[str]] = None,
                 print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): Features for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool, optional): Whether to print the evaluation results.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.

        Raises:
            ValueError: If the model has not been trained before evaluation.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        if eval_metrics is None:
            eval_metrics = ClassificationEvaluationMetrics.supported_metrics()

        # Ensure eval_metrics is a list
        if isinstance(eval_metrics, str):
            eval_metrics = [eval_metrics]

        probs = self.predict_proba(X)[:, 1]
        preds = self.predict(X)
        evaluation_results = {}
        for metric_name in eval_metrics:
            # translated_metric_name = translated_metrics.get(metric_name)
            metric_function = ClassificationEvaluationMetrics.get_metric(metric_name)
            if metric_function:
                evaluation_results[metric_name] = metric_function(y_true=y, y_prob=probs, y_pred=preds)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            self.print_evaluation_results(results=evaluation_results)
        return evaluation_results

    def plot_probability_distribution(self, X, y, save_path=None):
        """
        Plot the predicted probability distributions for each class in a binary classification model.

        Parameters:
        model : sklearn or similar binary classification model
            The trained binary classification model.
        X : array-like, shape (n_samples, n_features)
            The input samples.
        y : array-like, shape (n_samples,)
            The true labels.

        Returns:
        None
        """
        plt.clf()
        # Predict probabilities for each class
        probas = self.predict_proba(X)[:, 1]

        # Separate probabilities for each class
        class_0_probas = probas[y == 0]
        class_1_probas = probas[y == 1]

        # Plot the probability distributions
        # plt.figure()
        plt.hist(class_1_probas, bins=20, alpha=0.5, label='Class 1', color='blue')  # , density=1)
        plt.hist(class_0_probas, bins=20, alpha=0.5, label='Class 0', color='red')  # , density=1)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Frequency (%)')
        plt.title('Real Class Probability Distribution for Each Class')
        plt.legend(loc='upper center')
        # plt.gca().set_yticklabels(
        #     ['{:.0f}%'.format(x * 100) for x in plt.gca().get_yticks()])  # Format y-axis labels as percentages
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def plot_roc_curve(self, X, target, save_path=None):
        plt.clf()
        predictions = self.predict_proba(X)[:, 1]
        fpr, tpr, threshold = roc_curve(target, predictions)
        roc_auc = roc_auc_score(target, predictions)
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
                                  estimator_name=type(self).__name__)
        display.plot()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    def show_calibration(self, data, target, save_path=None):
        plt.clf()
        predicted_prob = self.predict_proba(data)[:, 1]
        CalibrationDisplay.from_predictions(target, predicted_prob)
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()

    @staticmethod
    def _optimal_threshold_auc(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]

    @staticmethod
    def _optimal_threshold_auprc(target, predicted):
        precision, recall, threshold = precision_recall_curve(target, predicted)
        # Remove last element
        precision = precision[:-1]
        recall = recall[:-1]

        i = np.arange(len(recall))
        roc = pd.DataFrame({'tf': pd.Series(precision * recall, index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]

        return list(roc_t['threshold'])[0]


class RegressionModel(Model):
    """
    Abstract base class for regression models, providing a framework for training and prediction in regression tasks.
    """

    @abstractmethod
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              training_parameters: Optional[Dict[str, Any]]) -> None:
        """
        Trains the regression model using provided training and validation data.

        Args:
            x_train (np.ndarray): observations for training.
            y_train (np.ndarray): Labels for training.
            training_parameters (Dict[str, Any], optional): Additional training parameters.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Makes predictions for the given input observations.

        Args:
            X (np.ndarray): observations for prediction.

        Returns:
            np.ndarray: The predicted values.

        Raises:
            NotImplementedError: Must be implemented by subclasses.
        """
        pass

    def evaluate(self, X: np.ndarray, y: np.ndarray, eval_metrics: List[str], print_results: bool = False) -> Dict[str, float]:
        """
        Evaluates the model using specified metrics.

        Args:
            X (np.ndarray): observations for evaluation.
            y (np.ndarray): True labels for evaluation.
            eval_metrics (List[str]): Metrics to use for evaluation.
            print_results (bool, optional): Whether to print the evaluation results.

        Returns:
            Dict[str, float]: A dictionary with metric names and their evaluated scores.

        Raises:
            ValueError: If the model has not been trained before evaluation.
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        predictions = self.predict(X)
        evaluation_results = {}

        for metric_name in eval_metrics:
            metric_function = RegressionEvaluationMetrics.get_metric(metric_name)
            if metric_function:
                evaluation_results[metric_name] = metric_function(y, predictions)
            else:
                print(f"Error: The metric '{metric_name}' is not supported.")

        if print_results:
            self.print_evaluation_results(results=evaluation_results)

        return evaluation_results

    def _ensure_numpy_arrays(self, observations: Any, labels: Optional[np.ndarray] = None) -> tuple:
        """
        Ensures that the input data is converted to NumPy array format, using the defined data preparation strategy.
        This method is used internally to standardize input data before training, predicting, or evaluating.

        Args:
            observations (Any): observations data, which can be in various formats like lists, Pandas DataFrames, or already in NumPy arrays.
            labels (np.ndarray, optional): Labels data, similar to observations in that it can be in various formats. If labels are not provided,
                                        only observations are converted and returned.

        Returns:
            tuple: The observations and labels (if provided) as NumPy arrays. If labels are not provided, labels in the tuple will be None.
        """
        if not isinstance(observations, np.ndarray) or (labels is not None and not isinstance(labels, np.ndarray)):
            return self.data_preparation_strategy.execute(observations, labels)[:2]
        else:
            return observations, labels
