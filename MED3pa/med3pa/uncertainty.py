"""
This module handles the computation of uncertainty metrics. 
It defines an abstract base class ``UncertaintyMetric`` and concrete implementations such as ``AbsoluteError`` for calculating uncertainty based on the difference between predicted probabilities and actual outcomes. 
An ``UncertaintyCalculator`` class is provided, which allows users to specify which uncertainty metric to use, 
thereby facilitating the use of customized uncertainty metrics for different analytical needs.
"""
from abc import ABC, abstractmethod

import numpy as np


class UncertaintyMetric(ABC):
    """
    Abstract base class for uncertainty metrics. Defines the structure that all uncertainty metrics should follow.
    """
    @abstractmethod
    def calculate(x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the uncertainty metric based on input observations, predicted probabilities, and true labels.

        Args:
            x (np.ndarray): Input observations.
            predicted_prob (np.ndarray): Predicted probabilities by the model.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: An array of uncertainty values for each prediction.
        """
        pass


class AbsoluteError(UncertaintyMetric):
    """
    Concrete implementation of the UncertaintyMetric class using absolute error.
    """
    def calculate(x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates the absolute error between predicted probabilities and true labels, providing a measure of
        prediction accuracy.

        Args:
            x (np.ndarray): Input features (not used in this metric but included for interface consistency).
            predicted_prob (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: Absolute errors between predicted probabilities and true labels.
        """
        return 1 - np.abs(y_true - predicted_prob)


class UncertaintyCalculator:
    """
    Class for calculating uncertainty using a specified uncertainty metric.
    """
    metric_mapping = {
        'absolute_error': AbsoluteError,
    }

    def __init__(self, metric_name: str) -> None:
        """
        Initializes the UncertaintyCalculator with a specific uncertainty metric.

        Args:
            metric_name (str): The name of the uncertainty metric to use for calculations.
        """
        if metric_name not in self.metric_mapping:
            raise ValueError(f"Unrecognized metric name: {metric_name}. Available metrics: {list(self.metric_mapping.keys())}")
        
        self.metric = self.metric_mapping[metric_name]
    
    def calculate_uncertainty(self, x: np.ndarray, predicted_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Calculates uncertainty for a set of predictions using the configured uncertainty metric.

        Args:
            x (np.ndarray): Input features.
            predicted_prob (np.ndarray): Predicted probabilities.
            y_true (np.ndarray): True labels.

        Returns:
            np.ndarray: Uncertainty values for each prediction, computed using the specified metric.
        """
        return self.metric.calculate(x, predicted_prob, y_true)

    @classmethod
    def supported_metrics(cls) -> list:
        """
        Returns a list of supported uncertainty metrics.

        Returns:
            list: A list of strings representing the names of the supported uncertainty metrics.
        """
        return list(cls.metric_mapping.keys())