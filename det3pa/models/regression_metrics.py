from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class RegressionEvaluationMetric(ABC):
    """
    Abstract base class for regression evaluation metrics.
    """
    @abstractmethod
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculates the metric value.

        Parameters
        ----------
        y_true : np.ndarray
            True values.
        y_pred : np.ndarray
            Predicted values.
        sample_weight : np.ndarray, optional
            Sample weights.

        Returns
        -------
        float
            Calculated metric value.
        """
        pass

class MeanSquaredError(RegressionEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return mean_squared_error(y_true, y_pred, sample_weight=sample_weight)

class RootMeanSquaredError(RegressionEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return np.sqrt(mean_squared_error(y_true, y_pred, sample_weight=sample_weight))

class MeanAbsoluteError(RegressionEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return mean_absolute_error(y_true, y_pred, sample_weight=sample_weight)

class R2Score(RegressionEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return r2_score(y_true, y_pred, sample_weight=sample_weight)

regression_metrics_mappings = {
    'MSE': MeanSquaredError,
    'RMSE': RootMeanSquaredError,
    'MAE': MeanAbsoluteError,
    'R2': R2Score
}
