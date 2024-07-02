"""
The ``abstract_metrics.py`` module defines the ``EvaluationMetric`` abstract base class, 
providing a standard interface for calculating metric values for model evaluations. 
"""

from abc import ABC


class EvaluationMetric(ABC):
    """
    Abstract base class for all evaluation metrics. This class provides a standardized interface for calculating
    metric values across different types of tasks, ensuring consistency and reusability.
    """

    @classmethod
    def get_metric(cls, metric_name: str):
        """
        Get the metric function based on the metric name.

        Args:
            metric_name (str): The name of the metric.

        Returns:
            function: The function corresponding to the metric.
        """
        pass

    @classmethod
    def supported_metrics(cls) -> list:
        """
        Get a list of supported regression metrics.

        Returns:
            list: A list of supported regression metrics.
        """
        pass
