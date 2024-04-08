import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mlease import OutlierRemover

class TestOutlierRemover(unittest.TestCase):

    def setUp(self):
        X, _ = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
        self.data = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4'])
        self.outlier_remover = OutlierRemover()

    def test_fit_transform_removes_outliers(self):
        transformed_data = self.outlier_remover.fit_transform(self.data)
        self.assertLess(transformed_data.shape[0], self.data.shape[0])

    def test_transform_removes_outliers_without_changing_columns(self):
        transformed_data = self.outlier_remover.transform(self.data)
        self.assertEqual(transformed_data.shape[1], self.data.shape[1])

    def test_fit_does_not_change_data(self):
        original_data = self.data.copy()
        self.outlier_remover.fit(self.data)
        pd.testing.assert_frame_equal(self.data, original_data)

    def test_fit_without_fit_transform_or_transform_raises_error(self):
        with self.assertRaises(ValueError):
            self.outlier_remover.fit()

    def test_threshold_parameter_within_range(self):
        with self.assertRaises(ValueError):
            self.outlier_remover = OutlierRemover(threshold=1.5)

if __name__ == '__main__':
    unittest.main()

