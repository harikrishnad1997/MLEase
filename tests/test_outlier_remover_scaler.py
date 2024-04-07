import unittest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from mlease import OutlierRemoverScaler

class TestOutlierRemoverScaler(unittest.TestCase):

    def setUp(self):
        X, _ = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
        self.data = pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4'])
        self.outlier_remover = OutlierRemoverScaler()

    def test_fit_transform(self):
        transformed_data = self.outlier_remover.fit_transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

    def test_transform(self):
        transformed_data = self.outlier_remover.transform(self.data)
        self.assertEqual(transformed_data.shape, self.data.shape)

if __name__ == '__main__':
    unittest.main()
