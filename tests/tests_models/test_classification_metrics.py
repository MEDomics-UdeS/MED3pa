import unittest
import numpy as np
from MED3pa.models.classification_metrics import ClassificationEvaluationMetrics

class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0])
        self.y_score = np.array([0.1, 0.9, 0.4, 0.2])

    def test_accuracy(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Accuracy')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.75)
    
    def test_recall(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Recall')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.5)
    
    def test_roc_auc(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Auc')
        result = metric_function(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 1.0)
    
    def test_average_precision(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Auprc')
        result = metric_function(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 1.0, places=3)
    
    def test_matthews_corrcoef(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('MCC')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.577, places=3)
    
    def test_precision(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Precision')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_f1_score(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('F1Score')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.666666, places=3)
    
    def test_sensitivity(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Sensitivity')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.5)
    
    def test_specificity(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Specificity')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_ppv(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('PPV')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_npv(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('NPV')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.666666, places=3)
    
    def test_balanced_accuracy(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('BalancedAccuracy')
        result = metric_function(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.75)
    
    def test_log_loss(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('LogLoss')
        result = metric_function(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.337538, places=3)

    def test_empty_arrays(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Accuracy')
        result = metric_function(np.array([]), np.array([]))
        self.assertIsNone(result)

    def test_single_class_roc_auc(self):
        metric_function = ClassificationEvaluationMetrics.get_metric('Auc')
        result = metric_function(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()
