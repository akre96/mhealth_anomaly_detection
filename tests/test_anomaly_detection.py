import mhealth_anomaly_detection.anomaly_detection as ad
import pandas as pd
import numpy as np
import pytest

N_DAYS = 10
WINDOW_SIZE = 3
N_FEATURES = 4
FEATURES = {
    f"example_feature_{i}": 1 + np.arange(N_DAYS) * i
    for i in range(N_FEATURES)
}
TEST_DATA = pd.DataFrame(
    {"subject_id": ["test_1"] * N_DAYS, "study_day": range(N_DAYS), **FEATURES}
)


@pytest.fixture(params=[0, 1])
def anomaly_data(request):
    np.random.seed(1)
    test_data = TEST_DATA.copy()
    n_anomalies = request.param
    start = 5
    for i in range(N_FEATURES):
        test_data[f"example_feature_{i}"] = (
            np.random.normal(0, 0.1, N_DAYS) + start
        )
        if n_anomalies > 0:
            test_data.loc[test_data.index[-1], f"example_feature_{i}"] = 100
    yield n_anomalies, test_data


@pytest.fixture(params=[0, 1])
def anomaly_data_early(request):
    np.random.seed(1)
    test_data = TEST_DATA.copy()
    n_anomalies = request.param
    start = 5
    for i in range(N_FEATURES):
        test_data[f"example_feature_{i}"] = (
            np.random.normal(0, 0.1, N_DAYS) + start
        )
        if n_anomalies > 0:
            test_data.loc[
                test_data.index[WINDOW_SIZE * 2], f"example_feature_{i}"
            ] = 100
    yield n_anomalies, test_data


# All anomaly detectors
@pytest.fixture(
    params=[
        ad.BaseRollingAnomalyDetector,
        ad.PCARollingAnomalyDetector,
        ad.NMFRollingAnomalyDetector,
        ad.PCAGridRollingAnomalyDetector,
        ad.IFRollingAnomalyDetector,
        ad.SVMRollingAnomalyDetector,
    ]
)
def anomaly_detector_all(request):
    print(str(request.param))
    yield request.param(
        features=list(FEATURES.keys()),
        window_size=WINDOW_SIZE,
        max_missing_days=0,
    )


# Only anomaly detectors with reconstruction error
@pytest.fixture(
    params=[
        ad.BaseRollingAnomalyDetector,
        ad.PCARollingAnomalyDetector,
        ad.NMFRollingAnomalyDetector,
        ad.PCAGridRollingAnomalyDetector,
    ]
)
def anomaly_detector_re(request):
    print(str(request.param))
    yield request.param(
        features=list(FEATURES.keys()),
        window_size=WINDOW_SIZE,
        max_missing_days=0,
    )


def test_detectors_can_label(anomaly_detector_all, anomaly_data):
    n_anomalies, test_data = anomaly_data
    labels = anomaly_detector_all.labelAnomaly(test_data)

    # Ensure labels are boolean
    assert labels.dtype == bool
    # Ensure all days are labelled
    assert labels.dropna().shape[0] == N_DAYS


# Test that re based detector outputs expected type and shape
@pytest.mark.parametrize("remove_anom", [True, False])
def test_input_output_base(anomaly_detector_re, anomaly_data, remove_anom):
    n_anomalies, test_data = anomaly_data
    anomaly_detector_re.remove_past_anomalies = remove_anom
    re_df = anomaly_detector_re.getReconstructionError(test_data)

    # in re_df expect 2 column + 1 column per feature
    assert re_df.shape == (N_DAYS, N_FEATURES + 2)
    assert re_df["total_re"].isnull().sum() == WINDOW_SIZE


# Input data cannot be missing full row for a day
def test_input_output_base_missing_day(anomaly_detector_re):
    # drop a day of data
    test_data = TEST_DATA.copy().drop(2)

    error = False
    try:
        anomaly_detector_re.getReconstructionError(test_data)
    except ValueError:
        error = True

    assert error


def test_detects_anomaly(anomaly_detector_re, anomaly_data):
    n_anomalies, test_data = anomaly_data
    test_data["anomaly"] = anomaly_detector_re.labelAnomaly(test_data)
    re = anomaly_detector_re.getReconstructionError(test_data)
    test_data["total_re"] = re["total_re"]

    # Ensure reconstruction error is calculated for all possible days
    assert re.total_re.isnull().sum() == WINDOW_SIZE

    # Ensure proper number of anomalies detected
    print(test_data[["total_re", "anomaly"]])
    assert test_data.anomaly.sum() == n_anomalies


def test_detects_anomaly_early(anomaly_detector_re, anomaly_data_early):
    n_anomalies, test_data = anomaly_data_early
    test_data["anomaly"] = anomaly_detector_re.labelAnomaly(test_data)
    re = anomaly_detector_re.getReconstructionError(test_data)
    test_data["total_re"] = re["total_re"]

    # Ensure reconstruction error is calculated for all possible days
    assert re.total_re.isnull().sum() == WINDOW_SIZE

    # Ensure proper number of anomalies detected
    print(test_data[["total_re", "anomaly"]])
    assert test_data.anomaly.sum() == n_anomalies

    # Finds the right time of anomaly
    if n_anomalies:
        assert test_data.anomaly.iloc[WINDOW_SIZE * 2] == True


def test_input_output_pc_methods_na_value():
    # drop a day of data
    test_data = TEST_DATA.copy()

    # Set 1 value to NaN
    test_data[test_data.index[2], list(FEATURES.keys())[1]] = np.nan
    detectors = [
        ad.PCARollingAnomalyDetector(
            features=list(FEATURES.keys()),
            window_size=3,
            max_missing_days=0,
            n_components=3,
        ),
        ad.NMFRollingAnomalyDetector(
            features=list(FEATURES.keys()),
            window_size=3,
            max_missing_days=0,
            n_components=3,
        ),
    ]

    for detector in detectors:
        # Ignoring divide by 0 warnings for PCA
        with np.errstate(divide="ignore", invalid="ignore"):
            re_df = detector.getReconstructionError(test_data)
        components = detector.components

        # in re_df expect 2 column + 1 column per feature
        assert re_df.shape == (N_DAYS, N_FEATURES + 2)

        # Captured pca component expected shape
        assert components.shape == (N_DAYS, detector.n_components, N_FEATURES)
