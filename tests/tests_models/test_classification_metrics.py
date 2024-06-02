import unittest
import numpy as np
from det3pa.models.classification_metrics import *

class TestClassificationMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 1, 0])
        self.y_pred = np.array([0, 1, 0, 0])
        self.y_score = np.array([0.1, 0.9, 0.4, 0.2])

    def test_accuracy(self):
        metric = Accuracy
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.75)
    
    def test_recall(self):
        metric = Recall
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.5)
    
    def test_roc_auc(self):
        metric = RocAuc
        result = metric.calculate(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 1.0)
    
    def test_average_precision(self):
        metric = AveragePrecision
        result = metric.calculate(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 1.0, places=3)
    
    def test_matthews_corrcoef(self):
        metric = MatthewsCorrCoef
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.577, places=3)
    
    def test_precision(self):
        metric = Precision
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_f1_score(self):
        metric = F1Score
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.666666, places=3)
    
    def test_sensitivity(self):
        metric = Sensitivity
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.5)
    
    def test_specificity(self):
        metric = Specificity
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_ppv(self):
        metric = PPV
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 1.0)
    
    def test_npv(self):
        metric = NPV
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.666666, places=3)
    
    def test_balanced_accuracy(self):
        metric = BalancedAccuracy
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.75)
    
    def test_log_loss(self):
        metric = LogLoss
        result = metric.calculate(self.y_true, self.y_score)
        self.assertAlmostEqual(result, 0.337538, places=3)

    def test_empty_arrays(self):
        metric = Accuracy
        result = metric.calculate(np.array([]), np.array([]))
        self.assertTrue(np.isnan(result))

    def test_single_class_roc_auc(self):
        metric = RocAuc
        result = metric.calculate(np.array([1, 1, 1]), np.array([0.9, 0.8, 0.7]))
        self.assertTrue(np.isnan(result))

if __name__ == '__main__':
    unittest.main()
