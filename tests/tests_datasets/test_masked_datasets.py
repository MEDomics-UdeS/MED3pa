import unittest
import numpy as np
from MED3pa.datasets import MaskedDataset

class TestMaskedDataset(unittest.TestCase):
    # sets up common variables that are used across tests
    def setUp(self):
        self.features = np.array([[1, 2], [3, 4], [5, 6]])
        self.true_labels = np.array([0, 1, 0])
        self.dataset = MaskedDataset(self.features, self.true_labels)

    # tests the __getitem__ correctly returns the feature vector, pseudo label and true label for a given index
    def test_get_item(self):
        x, y_hat, y = self.dataset[0]
        np.testing.assert_array_equal(x, self.features[0])
        self.assertEqual(y, self.true_labels[0])
    
    # tests that the refine method correctly applies a mask to select specific data points.
    def test_refine(self):
        mask = np.array([True, False, True])
        new_length = self.dataset.refine(mask)
        self.assertEqual(new_length, 2)
        np.testing.assert_array_equal(self.dataset.get_observations(), self.features[mask])
    
    # tests that the sample method correctly samples a specified number of data points from the dataset.
    def test_sample(self):
        sampled_dataset = self.dataset.sample_uniform(2, seed=42)
        self.assertEqual(len(sampled_dataset), 2)
    
    # asserts that the pseudo labels are correctly derived from the pseudo probabilities using the specified threshold.
    def test_set_pseudo_probabilities(self):
        pseudo_probs = np.array([0.6, 0.4, 0.8])
        self.dataset.set_pseudo_probs_labels(pseudo_probs, threshold=0.5)
        np.testing.assert_array_equal(self.dataset.get_pseudo_labels(), [1, 0, 1])
    
    # asserts that the cloned dataset has the same features, true labels, and pseudo labels as the original dataset.
    def test_clone(self):
        cloned_dataset = self.dataset.clone()
        np.testing.assert_array_equal(cloned_dataset.get_observations(), self.dataset.get_observations())
        np.testing.assert_array_equal(cloned_dataset.get_true_labels(), self.dataset.get_true_labels())
        np.testing.assert_array_equal(cloned_dataset.get_pseudo_labels(), self.dataset.get_pseudo_labels())

if __name__ == '__main__':
    unittest.main()
