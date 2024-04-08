import pandas as pd
import numpy as np
import pytest
from mlease import Bucketizer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': ['a', 'b', 'c', 'd', 'e']
    })

def test_fit_transform(sample_data):
    bucketizer = Bucketizer(num_classes=3, use_qcut=True)
    transformed_data = bucketizer.fit_transform(sample_data)

    assert transformed_data['A'].dtype == 'category'
    assert transformed_data['B'].dtype == 'category'
    assert transformed_data.shape == sample_data.shape

def test_invalid_input(sample_data):
    with pytest.raises(ValueError):
        bucketizer = Bucketizer(num_classes=3, use_qcut=True)
        bucketizer.fit_transform([1, 2, 3, 4, 5])  # Invalid input type

def test_automatic_num_classes(sample_data):
    bucketizer = Bucketizer(automatic_num_classes=True, use_qcut=True)
    transformed_data = bucketizer.fit_transform(sample_data)

    assert transformed_data['A'].dtype == 'category'
    assert transformed_data['B'].dtype == 'category'
    assert transformed_data.shape == sample_data.shape

def test_custom_labels(sample_data):
    custom_labels = ['low', 'medium', 'high']
    bucketizer = Bucketizer(num_classes=3, use_qcut=True, custom_labels=custom_labels)
    transformed_data = bucketizer.fit_transform(sample_data)

    assert all(label in transformed_data['A'].unique() for label in custom_labels)
    assert all(label in transformed_data['B'].unique() for label in custom_labels)

def test_cut_binning(sample_data):
    bucketizer = Bucketizer(num_classes=3, use_qcut=False)
    transformed_data = bucketizer.fit_transform(sample_data)

    assert transformed_data['A'].dtype == 'category'
    assert transformed_data['B'].dtype == 'category'
    assert transformed_data.shape == sample_data.shape

def test_invalid_column_dtype(sample_data):
    invalid_data = sample_data.copy()
    invalid_data['A'] = invalid_data['A'].astype(str)

    with pytest.raises(ValueError):
        bucketizer = Bucketizer(num_classes=3, use_qcut=True)
        bucketizer.fit_transform(invalid_data)

def test_invalid_num_classes(sample_data):
    with pytest.raises(ValueError):
        bucketizer = Bucketizer(num_classes=0, use_qcut=True)
        bucketizer.fit_transform(sample_data)

    with pytest.raises(ValueError):
        bucketizer = Bucketizer(num_classes=-1, use_qcut=True)
        bucketizer.fit_transform(sample_data)

def test_invalid_binning_method(sample_data):
    with pytest.raises(ValueError):
        bucketizer = Bucketizer(num_classes=3, use_qcut=None)
        bucketizer.fit_transform(sample_data)

def test_automatic_num_classes_calculation(sample_data):
    bucketizer = Bucketizer(automatic_num_classes=True, use_qcut=True)
    transformed_data = bucketizer.fit_transform(sample_data)

    assert transformed_data['A'].dtype == 'category'
    assert transformed_data['B'].dtype == 'category'
    assert transformed_data.shape == sample_data.shape


