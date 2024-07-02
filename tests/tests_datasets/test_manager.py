import unittest
import numpy as np
from MED3pa.datasets import DatasetsManager
from unittest.mock import patch

class TestDatasetsManager(unittest.TestCase):
    def setUp(self):
        self.manager = DatasetsManager()
        self.features = np.array([[1, 2], [3, 4], [5, 6]])
        self.true_labels = np.array([0, 1, 0])
        
        # Mock the data loading context
        self.patcher = patch('MED3pa.datasets.loading_context.DataLoadingContext.load_as_np', return_value=(['feature1', 'feature2'], self.features, self.true_labels))
        self.mock_load = self.patcher.start()
    
    def tearDown(self):
        self.patcher.stop()
    
    def test_set_training_data(self):
        self.manager.set_from_file('training','training.csv', 'target')
        features, labels = self.manager.get_dataset_by_type('training')
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(labels, self.true_labels)
    
    def test_set_validation_data(self):
        self.manager.set_from_file('validation','validation.csv', 'target')
        features, labels = self.manager.get_dataset_by_type('validation')
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(labels, self.true_labels)
    
    def test_set_reference_data(self):
        self.manager.set_from_file('reference','reference.csv', 'target')
        features, labels = self.manager.get_dataset_by_type('reference')
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(labels, self.true_labels)
    
    def test_set_testing_data(self):
        self.manager.set_from_file('testing','testing.csv', 'target')
        features, labels = self.manager.get_dataset_by_type('testing')
        np.testing.assert_array_equal(features, self.features)
        np.testing.assert_array_equal(labels, self.true_labels)
    
    def test_column_label_mismatch(self):
        # Set the training data with consistent column labels
        self.manager.set_from_file('training','training.csv', 'target')
        
        # Now, mock the load_as_np to return different column labels for the validation call
        with patch('MED3pa.datasets.loading_context.DataLoadingContext.load_as_np', return_value=(['feature1', 'feature2', 'extra'], self.features, self.true_labels)):
            with self.assertRaises(ValueError):
                self.manager.set_from_file('validation', 'validation.csv', 'target')

if __name__ == '__main__':
    unittest.main()
