import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RegressionEvaluationMetrics:
    """
    A class to compute various regression evaluation metrics.
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Mean Squared Error (MSE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Mean Squared Error.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Root Mean Squared Error (RMSE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Root Mean Squared Error.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Mean Absolute Error (MAE).

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Mean Absolute Error.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the R-squared (R2) score.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: R-squared score.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return r2_score(y_true, y_pred, sample_weight=sample_weight)
    
    @classmethod
    def get_metric(cls, metric_name: str):
        """
        Get the metric function based on the metric name.

        Args:
            metric_name (str): The name of the metric.

        Returns:
            function: The function corresponding to the metric.
        """
        metrics_mappings = {
            'MSE': cls.mean_squared_error,
            'RMSE': cls.root_mean_squared_error,
            'MAE': cls.mean_absolute_error,
            'R2': cls.r2_score
        }
        return metrics_mappings.get(metric_name)
    
    @classmethod
    def supported_metrics(cls) -> list:
        """
        Get a list of supported regression metrics.

        Returns:
            list: A list of supported regression metrics.
        """
        return list(cls.get_metric.keys())


