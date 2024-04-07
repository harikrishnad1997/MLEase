import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from mlease import MissingValueImputer

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, np.nan, 30, 40, 50],
        'C': [100, 200, 300, 400, 500],
        'D': ['a', 'b', 'c', 'd', 'e']
    })

def test_mean_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='mean')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()
    assert np.allclose(data_imputed['A'].mean(), data_imputed['A'].fillna(0).mean())
    assert np.allclose(data_imputed['B'].mean(), data_imputed['B'].fillna(0).mean())

def test_median_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='median')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()
    assert np.allclose(data_imputed['A'].median(), data_imputed['A'].fillna(0).median())
    assert np.allclose(data_imputed['B'].median(), data_imputed['B'].fillna(0).median())

def test_most_frequent_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='most_frequent')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()
    assert data_imputed['D'].mode().iloc[0] == data_imputed['D'].fillna('').mode().iloc[0]

def test_group_wise_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='group_wise', group_cols=['D'])
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()
    assert np.allclose(data_imputed.groupby('D')['A'].transform('mean'), data_imputed['A'])
    assert np.allclose(data_imputed.groupby('D')['B'].transform('mean'), data_imputed['B'])

def test_time_series_spline_imputation(sample_data):
    sample_data['time'] = pd.date_range(start='2023-01-01', periods=5, freq='D')
    imputer = MissingValueImputer(imputation_method='time_series_spline', time_col='time')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()

def test_time_series_linear_imputation(sample_data):
    sample_data['time'] = pd.date_range(start='2023-01-01', periods=5, freq='D')
    imputer = MissingValueImputer(imputation_method='time_series_linear', time_col='time')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()

def test_iterative_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='iterative')
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()

def test_prediction_imputation(sample_data):
    imputer = MissingValueImputer(imputation_method='prediction', prediction_model=LinearRegression())
    data_imputed = imputer.fit_transform(sample_data)

    assert not data_imputed.isnull().values.any()

def test_invalid_imputation_method(sample_data):
    with pytest.raises(ValueError):
        imputer = MissingValueImputer(imputation_method='invalid')
        imputer.fit_transform(sample_data)