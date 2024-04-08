import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from mlease import OutlierRemover
import pytest

@pytest.fixture
def data():
    X, _ = make_classification(n_samples=1000, n_features=4, n_classes=2, random_state=42)
    return pd.DataFrame(X, columns=['feat1', 'feat2', 'feat3', 'feat4'])

@pytest.fixture
def outlier_remover():
    return OutlierRemover()

def test_fit_transform_removes_outliers(data, outlier_remover):
    transformed_data = outlier_remover.fit_transform(data)
    assert transformed_data.shape[0] < data.shape[0]

def test_transform_removes_outliers_without_changing_columns(data, outlier_remover):
    transformed_data = outlier_remover.transform(data)
    assert transformed_data.shape[1] == data.shape[1]

def test_fit_does_not_change_data(data, outlier_remover):
    original_data = data.copy()
    outlier_remover.fit(data)
    pd.testing.assert_frame_equal(data, original_data)

def test_fit_without_fit_transform_or_transform_raises_error(outlier_remover):
    with pytest.raises(ValueError):
        outlier_remover.fit()

def test_threshold_parameter_within_range():
    with pytest.raises(ValueError):
        OutlierRemover(threshold=1.5)
