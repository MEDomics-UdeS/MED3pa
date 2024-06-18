"""
The ``abstract_metrics.py`` module defines the ``EvaluationMetric`` abstract base class, 
providing a standard interface for calculating metric values for model evaluations. 
It includes specialized subclasses like ``ClassificationEvaluationMetric`` and ``RegressionEvaluationMetric`` to cater to specific needs of classification and regression tasks, 
facilitating the integration and extension of new evaluation metrics."""

from abc import ABC, abstractmethod
import numpy as np

class EvaluationMetric(ABC):
    """
    Abstract base class for all evaluation metrics. This class provides a standardized interface for calculating
    metric values across different types of tasks, ensuring consistency and reusability.
    """

    @abstractmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculates the metric value based on true labels/values and predicted labels/values.

        Args:
            y_true (np.ndarray): True labels or values, representing the ground truth data.
            y_pred (np.ndarray): Predicted scores, labels, or values, as produced by a model.
            sample_weight (np.ndarray, optional): Array of weights that are assigned to individual
                                                  samples, mainly used in averaging the metric.

        Returns:
            float: The calculated metric value.
        """
        pass

class ClassificationEvaluationMetric(EvaluationMetric):
    """
    Base class for classification metrics.
    """
    pass

class RegressionEvaluationMetric(EvaluationMetric):
    """
    Base class for regression metrics.
    """
    pass