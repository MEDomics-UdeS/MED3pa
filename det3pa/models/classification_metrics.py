"""
This module defines a comprehensive list of Evaluation metrics for classification tasks.
"""
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             average_precision_score, matthews_corrcoef,
                             precision_score, f1_score, log_loss)
from .abstract_metrics import ClassificationEvaluationMetric


class Accuracy(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return accuracy_score(y_true, y_pred, sample_weight=sample_weight)

class Recall(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return recall_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

class RocAuc(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if len(np.unique(y_true)) == 1:
            return np.nan
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return roc_auc_score(y_true, y_pred, sample_weight=sample_weight)

class AveragePrecision(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return average_precision_score(y_true, y_pred, sample_weight=sample_weight)

class MatthewsCorrCoef(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return matthews_corrcoef(y_true, y_pred, sample_weight=sample_weight)

class Precision(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return precision_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

class F1Score(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return f1_score(y_true, y_pred, sample_weight=sample_weight, zero_division=0)

class Sensitivity(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return recall_score(y_true, y_pred, pos_label=1, zero_division=0)

class Specificity(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return recall_score(y_true, y_pred, pos_label=0, zero_division=0)

class PPV(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return precision_score(y_true, y_pred, pos_label=1, zero_division=0)

class NPV(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        return precision_score(y_true, y_pred, pos_label=0, zero_division=0)

class BalancedAccuracy(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        sens = Sensitivity.calculate(y_true, y_pred)
        spec = Specificity.calculate(y_true, y_pred)
        return (sens + spec) / 2

class LogLoss(ClassificationEvaluationMetric):
    def calculate(y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray = None) -> float:
        if y_true.size == 0 or y_pred.size == 0:
            return np.nan
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return log_loss(y_true, y_pred, sample_weight=sample_weight)

metrics_mappings = {
    'Accuracy': Accuracy,
    'BalancedAccuracy': BalancedAccuracy,
    'Precision': Precision,
    'Recall': Recall,
    'F1Score': F1Score,
    'Specificity': Specificity,
    'Sensitivity': Sensitivity,
    'Auc': RocAuc,
    'LogLoss': LogLoss,
    'Auprc': AveragePrecision,
    'NPV': NPV,
    'PPV': PPV,
    'MCC': MatthewsCorrCoef
}
