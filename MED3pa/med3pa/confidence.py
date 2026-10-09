"""This module handles the computation of predictive confidence metrics, the targets of the IPC, APC and MPC models.
It defines an abstract base class ``ConfidenceMetric`` and concrete implementations such as ``AbsoluteError`` for
calculating confidence based on the difference between predicted probabilities and actual outcomes. A
``ConfidenceCalculator`` class is provided, which allows users to specify which confidence metric to use, thereby
facilitating the use of customized confidence metrics for different analytical needs."""

import numpy as np
from abc import ABC, abstractmethod


class ConfidenceMetric(ABC):
    """
    Abstract base class for confidence metrics. Defines the structure that all confidence metrics should follow.
    """
    @staticmethod
    @abstractmethod
    def calculate(x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the confidence metric based on input observations, predicted probabilities, and true labels.

        Args:
            x (np.ndarray): Input observations.
            predicted_prob (np.ndarray): Predicted probabilities by the model.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: An array of confidence values for each prediction.
        """
        pass


class AbsoluteError(ConfidenceMetric):
    """
    Concrete implementation of the ConfidenceMetric class using absolute error.
    """
    @staticmethod
    def calculate(x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the confidence as one minus the absolute error between predicted probabilities and true labels,
        providing a measure of prediction accuracy.

        Args:
            x (np.ndarray): Input features (not used in this metric but included for interface consistency).
            predicted_prob (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: One minus the absolute errors between predicted probabilities and true labels.
        """
        return 1 - np.abs(y_true - predicted_prob)


class SigmoidalError(ConfidenceMetric):
    """
    Concrete implementation of the ConfidenceMetric class using Sigmoidal error.
    """
    @staticmethod
    def calculate(x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray, threshold=0.5) -> np.ndarray:
        """
        Calculates the confidence as a sigmoidal function of the error between predicted probabilities and true
        labels, providing a measure of prediction accuracy.

        Args:
            x (np.ndarray): Input features (not used in this metric but included for interface consistency).
            predicted_prob (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels.
            threshold (float): Classification threshold

        Returns:
            np.ndarray: Sigmoidal confidence values of the predicted probabilities.
        """
        return 1 / (
                1 + np.exp(10 * np.log(3) * (np.abs(y_true - predicted_prob) - np.abs(threshold - y_true))))


class ConfidenceCalculator:
    """
    Class for calculating predictive confidence using a specified confidence metric.
    """
    metric_mapping = {
        'absolute_error': AbsoluteError,
        'sigmoidal_error': SigmoidalError,
    }

    def __init__(self, metric_name: str) -> None:
        """
        Initializes the ConfidenceCalculator with a specific confidence metric.

        Args:
            metric_name (str): The name of the confidence metric to use for calculations.
        """
        if metric_name not in self.metric_mapping:
            raise ValueError(f"Unrecognized metric name: {metric_name}. Available metrics: {list(self.metric_mapping)}")

        self.metric = self.metric_mapping[metric_name]

    def calculate_confidence(self, x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates confidence for a set of predictions using the configured confidence metric.

        Args:
            x (np.ndarray): Input features.
            predicted_prob (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: Confidence values for each prediction, computed using the specified metric.
        """
        return self.metric.calculate(x, predicted_prob, y_true)

    @classmethod
    def supported_metrics(cls) -> list:
        """
        Returns a list of supported confidence metrics.

        Returns:
            list: A list of strings representing the names of the supported confidence metrics.
        """
        return list(cls.metric_mapping)
