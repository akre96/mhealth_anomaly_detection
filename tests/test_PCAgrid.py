from mhealth_anomaly_detection.PCAgrid import PCAgrid
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, strategies as st


# Baseline parameters
N_COMPONENTS = 2
N_DAYS = 10
N_FEATURES = 4


def getData(n_days, n_features):
    np.random.seed(42)
    baseline = 10

    return (
        np.random.normal(scale=0.01, loc=0, size=(n_days, n_features))
        + baseline
    )


@pytest.fixture(params=[10, 20, 30])
def n_days(request):
    return request.param


@pytest.fixture(params=[4, 5, 6])
def n_features(request):
    return request.param


@pytest.fixture(params=[2, 3, 4])
def n_components(request):
    return request.param


def test_loads_simple(n_components):
    pca = PCAgrid(n_components=n_components)
    assert pca.trained == False


def test_fit_static(n_days, n_features, n_components):
    X = np.zeros((n_days, n_features))
    pca = PCAgrid(n_components=n_components)
    pca.fit(X)
    assert pca.trained == True


def test_fit_simple():
    X = getData(N_DAYS, N_FEATURES)
    pca = PCAgrid(n_components=N_COMPONENTS)
    pca.fit(X)
    assert pca.trained == True


def test_transform_simple():
    X = getData(N_DAYS, N_FEATURES)
    pca = PCAgrid(n_components=N_COMPONENTS)
    pca.fit(X)
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (N_DAYS, N_COMPONENTS)


def test_inverse_transform(n_days, n_features, n_components):
    X = getData(n_days, n_features)
    pca = PCAgrid(n_components=n_components)
    pca.fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    assert X_reconstructed.shape == (n_days, n_features)
    assert np.allclose(X, X_reconstructed, atol=1e-1)

    # Ensure works on pandas DataFrame
    X_pd = pd.DataFrame(X)
    pca.fit(pd.DataFrame(X_pd))
    X_transformed = pca.transform(X_pd)
    X_reconstructed = pca.inverse_transform(X_transformed)

    assert X_reconstructed.shape == (n_days, n_features)
    assert np.allclose(X, X_reconstructed, atol=1e-1)
