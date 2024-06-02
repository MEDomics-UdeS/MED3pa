import unittest
import numpy as np
from det3pa.models.regression_metrics import MeanSquaredError, MeanAbsoluteError, R2Score

class TestRegressionMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([1.0, 2.0, 3.0, 4.0])
        self.y_pred = np.array([1.1, 1.9, 3.2, 3.9])
    
    def test_mean_squared_error(self):
        metric = MeanSquaredError
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.0175, places=3)
    
    def test_mean_absolute_error(self):
        metric = MeanAbsoluteError
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.125, places=3)
    
    def test_r2_score(self):
        metric = R2Score
        result = metric.calculate(self.y_true, self.y_pred)
        self.assertAlmostEqual(result, 0.986, places=3)

    def test_empty_arrays(self):
        metric = MeanSquaredError
        result = metric.calculate(np.array([]), np.array([]))
        self.assertTrue(np.isnan(result))

    def test_identical_arrays(self):
        metric = MeanSquaredError
        result = metric.calculate(self.y_true, self.y_true)
        self.assertAlmostEqual(result, 0.0, places=3)

if __name__ == '__main__':
    unittest.main()
