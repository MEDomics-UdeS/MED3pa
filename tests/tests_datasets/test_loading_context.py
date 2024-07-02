import unittest
from MED3pa.datasets import DataLoadingContext
from MED3pa.datasets import CSVDataLoadingStrategy

class TestDataLoadingContext(unittest.TestCase):
    # tests that the correct loading strategy is selected
    def test_csv_strategy(self):
        ctx = DataLoadingContext('data.csv')
        self.assertIsInstance(ctx.get_strategy(), CSVDataLoadingStrategy)
    
    # tests that the ValueError is raised when given an unsupported extension file
    def test_invalid_strategy(self):
        with self.assertRaises(ValueError):
            DataLoadingContext('data.invalid')

if __name__ == '__main__':
    unittest.main()
