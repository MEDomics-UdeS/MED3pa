import unittest
import numpy as np
from MED3pa.datasets import CSVDataLoadingStrategy

class TestCSVDataLoadingStrategy(unittest.TestCase):
    # verify that CSVDataLoadingStrategy correctly loads data from a CSV file and separates it into features and target arrays.
    def test_execute(self):
        # Create a temporary CSV file for testing
        import tempfile
        import pandas as pd
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp:
            df = pd.DataFrame({
                'feature1': [1, 2, 3],
                'feature2': [4, 5, 6],
                'target': [0, 1, 0]
            })
            df.to_csv(tmp.name, index=False)
            column_labels, features_np, target_np = CSVDataLoadingStrategy.execute(tmp.name, 'target')
            
            self.assertEqual(column_labels, ['feature1', 'feature2'])
            np.testing.assert_array_equal(features_np, df[['feature1', 'feature2']].values)
            np.testing.assert_array_equal(target_np, df['target'].values)

if __name__ == '__main__':
    unittest.main()
