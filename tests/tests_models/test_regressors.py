import unittest
import numpy as np
from MED3pa.models.concrete_regressors import RandomForestRegressorModel, DecisionTreeRegressorModel

class TestRandomForestRegressorModel(unittest.TestCase):
    def setUp(self):
        self.model = RandomForestRegressorModel(params={"n_estimators": 10})
        self.features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.labels = np.array([1.0, 2.0, 3.0, 4.0])
        self.validation_features = np.array([[2, 3], [4, 5]])
        self.validation_labels = np.array([2.5, 3.5])
    
    def test_train(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        predictions = self.model.predict(self.features)
        self.assertEqual(predictions.shape, self.labels.shape)
    
    def test_evaluate(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        results = self.model.evaluate(self.validation_features, self.validation_labels, ['MSE'], print_results=False)
        self.assertIn('MSE', results)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, {"invalid_param": 1})

    def test_empty_train_data(self):
        with self.assertRaises(ValueError):
            self.model.train(np.array([]), np.array([]), self.validation_features, self.validation_labels, None)

class TestDecisionTreeRegressorModel(unittest.TestCase):
    def setUp(self):
        self.model = DecisionTreeRegressorModel(params={"max_depth": 3})
        self.features = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        self.labels = np.array([1.0, 2.0, 3.0, 4.0])
        self.validation_features = np.array([[2, 3], [4, 5]])
        self.validation_labels = np.array([2.5, 3.5])
    
    def test_train(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        self.assertIsNotNone(self.model.model)
    
    def test_predict(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        predictions = self.model.predict(self.features)
        self.assertEqual(predictions.shape, self.labels.shape)
    
    def test_evaluate(self):
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, None)
        results = self.model.evaluate(self.validation_features, self.validation_labels, ['MSE'], print_results=False)
        self.assertIn('MSE', results)

    def test_invalid_params(self):
        with self.assertRaises(ValueError):
            self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, {"invalid_param": 1})

    def test_empty_train_data(self):
        with self.assertRaises(ValueError):
            self.model.train(np.array([]), np.array([]), self.validation_features, self.validation_labels, None)

if __name__ == '__main__':
    unittest.main()
