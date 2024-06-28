from typing import List

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, log_loss, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)


class ClassificationEvaluationMetrics:
    """
    A class to compute various classification evaluation metrics.
    """
    
    @staticmethod
    def accuracy(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the accuracy score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Accuracy score.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def recall(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the recall score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Recall score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    
    @staticmethod
    def roc_auc(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the ROC AUC score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted probabilities.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: ROC AUC score.
        """
        if len(np.unique(y_true)) == 1 or len(np.unique(y_true)) == 1:
            return np.nan
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return roc_auc_score(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def average_precision(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the average precision score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted probabilities.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Average precision score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return average_precision_score(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def matthews_corrcoef(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the Matthews correlation coefficient.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Matthews correlation coefficient.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)
    
    @staticmethod
    def precision(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the precision score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Precision score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    
    @staticmethod
    def f1_score(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the F1 score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: F1 score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)
    
    @staticmethod
    def sensitivity(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the sensitivity (recall for the positive class).

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Sensitivity score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    @staticmethod
    def specificity(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the specificity (recall for the negative class).

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Specificity score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    @staticmethod
    def ppv(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the positive predictive value (PPV).

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Positive predictive value.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    
    @staticmethod
    def npv(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the negative predictive value (NPV).

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Negative predictive value.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        return precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    
    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the balanced accuracy score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Balanced accuracy score.
        """
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        sens = ClassificationEvaluationMetrics.sensitivity(y_true, y_pred)
        spec = ClassificationEvaluationMetrics.specificity(y_true, y_pred)
        return (sens + spec) / 2
    
    @staticmethod
    def log_loss(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        """
        Calculate the log loss score.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted probabilities.
            sample_weight (np.ndarray, optional): Sample weights.

        Returns:
            float: Log loss score.
        """
        if y_true.size == 0 or y_pred.size == 0 or len(np.unique(y_true)) == 1:
            return np.nan
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred, sample_weight=sample_weight)
    
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
            'Accuracy': cls.accuracy,
            'BalancedAccuracy': cls.balanced_accuracy,
            'Precision': cls.precision,
            'Recall': cls.recall,
            'F1Score': cls.f1_score,
            'Specificity': cls.specificity,
            'Sensitivity': cls.sensitivity,
            'Auc': cls.roc_auc,
            'LogLoss': cls.log_loss,
            'Auprc': cls.average_precision,
            'NPV': cls.npv,
            'PPV': cls.ppv,
            'MCC': cls.matthews_corrcoef
        }
        return metrics_mappings.get(metric_name)
    
    @classmethod
    def supported_metrics(cls) -> List[str]:
        """
        Get a list of supported classification metrics.

        Returns:
            list: A list of supported classification metrics.
        """
        return list(cls.get_metric.keys())

