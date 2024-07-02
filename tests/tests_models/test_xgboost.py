import unittest
import numpy as np
import xgboost as xgb
from unittest.mock import patch, MagicMock
from MED3pa.models.concrete_classifiers import XGBoostModel


class TestXGBoostModel(unittest.TestCase):
    
    def setUp(self):
        self.params = {
            'booster': 'gbtree',
            'objective': 'binary:logistic',
            'eval_metric': 'logloss'
        }
        self.model = XGBoostModel(params=self.params)
        self.features = np.array([[1, 2], [3, 4], [5, 6]])
        self.labels = np.array([0, 1, 0])
        self.validation_features = np.array([[1, 2], [3, 4]])
        self.validation_labels = np.array([0, 1])
    
    def test_initialize_model_with_params(self):
        self.assertIsNotNone(self.model.params)
        self.assertIsInstance(self.model, XGBoostModel)
    
    def test_train_model(self):
        training_parameters = {
            'custom_eval_metrics': ['Accuracy', 'Precision'],
            'num_boost_rounds': 10
        }
        self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, training_parameters, False)
        trained_model = self.model.get_model()
        self.assertTrue(trained_model is not None)
    
    def test_predict_with_trained_model(self):
        self.model.set_model(xgb.Booster())
        dtrain = xgb.DMatrix(self.features, label=self.labels)
        self.model.model = xgb.train({'objective': 'binary:logistic'}, dtrain)
        predictions = self.model.predict(self.features)
        self.assertIsNotNone(predictions)
        self.assertEqual(predictions.shape[0], self.features.shape[0])

    def test_evaluate_model(self):
        with patch.object(self.model, '_ensure_dmatrix', return_value=xgb.DMatrix(self.features, self.labels)):
            with patch.object(xgb.Booster, 'predict', return_value=np.array([0.1, 0.4, 0.8])):
                self.model.model = xgb.Booster({'booster': 'gbtree'})
                self.model.model_class = xgb.Booster
                evaluation_metrics = ['Accuracy', 'LogLoss']
                results = self.model.evaluate(self.features, self.labels, evaluation_metrics)
                self.assertIn('Accuracy', results)
                self.assertIn('LogLoss', results)
    
    def test_train_to_disagree(self):
        training_parameters = {
            'custom_eval_metrics': ['Accuracy', 'Precision'],
            'num_boost_rounds': 10
        }
        self.model.train_to_disagree(self.features, self.labels, self.validation_features, self.validation_labels, self.features, self.labels, training_parameters, False, 3)
        self.assertTrue(self.model.model is not None)
    
    def test_set_model(self):
        booster = xgb.Booster({'booster': 'gbtree'})
        self.model.set_model(booster)
        self.assertEqual(self.model.model, booster)
        self.assertEqual(self.model.model_class, xgb.Booster)

    def test_train_with_empty_data(self):
        empty_features = np.array([])
        empty_labels = np.array([])
        with self.assertRaises(ValueError):
            self.model.train(empty_features, empty_labels, self.validation_features, self.validation_labels, {}, False)

    def test_predict_with_uninitialized_model(self):
        uninitialized_model = XGBoostModel(params=self.params)
        with self.assertRaises(ValueError):
            uninitialized_model.predict(self.features)

    def test_evaluate_with_empty_data(self):
        self.model.model = xgb.Booster({'booster': 'gbtree'})
        self.model.model_class = xgb.Booster
        with self.assertRaises(ValueError):
            self.model.evaluate(np.array([]), np.array([]), ['Accuracy'])
    
    def test_train_with_invalid_params(self):
        invalid_params = {'invalid_param': 'invalid_value'}
        with self.assertRaises(ValueError):
            self.model.train(self.features, self.labels, self.validation_features, self.validation_labels, invalid_params, False)
    
    def test_evaluate_with_unsupported_metrics(self):
        with patch.object(self.model, '_ensure_dmatrix', return_value=xgb.DMatrix(self.features, self.labels)):
            with patch.object(xgb.Booster, 'predict', return_value=np.array([0.1, 0.4, 0.8])):
                self.model.model = xgb.Booster({'booster': 'gbtree'})
                self.model.model_class = xgb.Booster
                unsupported_metrics = ['UnsupportedMetric']
                with self.assertRaises(ValueError):
                    self.model.evaluate(self.features, self.labels, unsupported_metrics)

    

if __name__ == '__main__':
    unittest.main()
