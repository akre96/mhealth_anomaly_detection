from mhealth_anomaly_detection.PCAgrid import PCAgrid
import numpy as np


def test_loads():
    pca = PCAgrid()
    assert pca.trained == False


def test_fit():
    pca = PCAgrid()
    X = np.random.rand(100, 10)
    pca.fit(X)
    assert pca.trained == True


def test_transform():
    pca = PCAgrid()
    X = np.random.rand(100, 10)
    pca.fit(X)
    X_transformed = pca.transform(X)
    assert X_transformed.shape == (100, 2)


def test_inverse_transform():
    pca = PCAgrid()
    X = np.ones((100, 10))
    pca.fit(X)
    X_transformed = pca.transform(X)
    X_reconstructed = pca.inverse_transform(X_transformed)
    assert X_reconstructed.shape == (100, 10)
    assert np.allclose(X, X_reconstructed)
