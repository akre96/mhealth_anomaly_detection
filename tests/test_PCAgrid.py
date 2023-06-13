from mhealth_anomaly_detection.PCAgrid import PCAgrid
import numpy as np
import pandas as pd
import pytest

N_COMPONENTS = 2
N_DAYS = 10
N_FEATURES = 4


@pytest.fixture()
def X():
    np.random.seed(42)
    baseline = 10

    yield np.random.normal(
        scale=0.01, loc=0, size=(N_DAYS, N_FEATURES)
    ) + baseline


def test_loads():
    pca = PCAgrid(n_components=N_COMPONENTS)
    assert pca.trained == False


def test_fit(X):
    pca = PCAgrid(n_components=N_COMPONENTS)
    pca.fit(X)
    assert pca.trained == True


def test_transform(X):
    pca = PCAgrid(n_components=N_COMPONENTS)
    pca.fit(X)
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (N_DAYS, N_COMPONENTS)


def test_inverse_transform(X):
    pca = PCAgrid(n_components=N_COMPONENTS)
    pca.fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)

    assert X_reconstructed.shape == (N_DAYS, N_FEATURES)
    assert np.allclose(X, X_reconstructed, atol=1e-1)

    # Ensure works on pandas DataFrame
    X_pd = pd.DataFrame(X)
    pca.fit(pd.DataFrame(X_pd))
    X_transformed = pca.transform(X_pd)
    X_reconstructed = pca.inverse_transform(X_transformed)

    assert X_reconstructed.shape == (N_DAYS, N_FEATURES)
    assert np.allclose(X, X_reconstructed, atol=1e-1)
