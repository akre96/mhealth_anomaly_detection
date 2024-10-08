import mhealth_anomaly_detection.anomaly_performance_metrics as apm
import pandas as pd
import numpy as np

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


def test_performance_metrics_perfect():
    with_anomalies = TEST_DATA.copy()

    # All True positives
    with_anomalies["anomaly"] = 1
    anomaly_detector_cols = ["model_anomaly"]
    with_anomalies["model_anomaly"] = 1
    groupby_cols = ["subject_id"]

    metrics = apm.binaryPerformanceMetrics(
        data=with_anomalies,
        anomaly_detector_cols=anomaly_detector_cols,
        groupby_cols=groupby_cols,
    )
    print(
        metrics[
            [
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
            ]
        ]
    )
    print(
        metrics[
            [
                "accuracy",
                "sensitivity",
                "specificity",
            ]
        ]
    )

    # columns:
    #   Groupby col, sens, spec, accuracy, TN, TP, FN, FP, F1, Recall
    assert metrics.shape == (1, 11)

    assert metrics.true_positives.sum() == N_DAYS
    assert metrics.true_negatives.sum() == 0
    assert metrics.false_negatives.sum() == 0
    assert metrics.false_positives.sum() == 0

    # Perfect performance expected
    assert np.all(metrics.sensitivity == 1.0)
    assert np.all(np.isnan(metrics.specificity))
    assert np.all(metrics.accuracy == 1.0)


def test_performance_metrics_worse():
    with_anomalies = TEST_DATA.copy()

    # All True positives
    with_anomalies["anomaly"] = 1
    anomaly_detector_cols = ["model_anomaly"]
    with_anomalies["model_anomaly"] = 0
    groupby_cols = ["subject_id"]

    metrics = apm.binaryPerformanceMetrics(
        data=with_anomalies,
        anomaly_detector_cols=anomaly_detector_cols,
        groupby_cols=groupby_cols,
    )

    print(
        metrics[
            [
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
            ]
        ]
    )
    print(
        metrics[
            [
                "accuracy",
                "sensitivity",
                "specificity",
            ]
        ]
    )

    assert metrics.true_positives.sum() == 0
    assert metrics.true_negatives.sum() == 0
    assert metrics.false_negatives.sum() == N_DAYS
    assert metrics.false_positives.sum() == 0

    # Poor performance expected
    assert np.all(metrics.sensitivity == 0)
    assert np.all(np.isnan(metrics.specificity))
    assert np.all(metrics.accuracy == 0)


def test_performance_metrics_mid():
    # Fix to 10 days
    N_DAYS = 10
    N_FEATURES = 3
    FEATURES = {
        f"example_feature_{i}": np.arange(N_DAYS) * i
        for i in range(N_FEATURES)
    }
    TEST_DATA = pd.DataFrame(
        {
            "subject_id": ["test_1"] * N_DAYS,
            "study_day": range(N_DAYS),
            **FEATURES,
        }
    )
    with_anomalies = TEST_DATA.copy()

    # Half True Positives and False Positives
    with_anomalies["anomaly"] = 1
    with_anomalies.loc[with_anomalies.study_day >= 5, "anomaly"] = 0
    anomaly_detector_cols = ["model_anomaly"]
    with_anomalies["model_anomaly"] = 1
    groupby_cols = ["subject_id"]

    metrics = apm.binaryPerformanceMetrics(
        data=with_anomalies,
        anomaly_detector_cols=anomaly_detector_cols,
        groupby_cols=groupby_cols,
    )
    print(
        metrics[
            [
                "false_positives",
                "false_negatives",
                "true_positives",
                "true_negatives",
            ]
        ]
    )
    print(
        metrics[
            [
                "accuracy",
                "sensitivity",
                "specificity",
            ]
        ]
    )

    # columns:
    #   Groupby col, sens, spec, accuracy, TN, TP, FN, FP
    assert metrics.shape == (1, 8 + N_FEATURES)

    assert metrics.true_positives.sum() == 5
    assert metrics.true_negatives.sum() == 0
    assert metrics.false_negatives.sum() == 0
    assert metrics.false_positives.sum() == 5

    # Perfect performance expected
    assert np.all(metrics.sensitivity == 1.0)
    assert np.all(metrics.specificity == 0)
    assert np.all(metrics.accuracy == 0.5)


# distance dataframe should be empty if no real anomalies
def test_distance_empty():
    with_anomalies = TEST_DATA.copy()
    anomaly_detector_cols = ["model_anomaly"]
    groupby_cols = ["subject_id"]

    # Test no anomalies detected
    with_anomalies["anomaly"] = 0
    with_anomalies["model_anomaly"] = 1

    distance = apm.distance_real_to_detected_anomaly(
        data=with_anomalies,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )
    print(distance)
    assert distance.empty


def test_distance_perfect():
    with_anomalies = TEST_DATA.copy()
    anomaly_detector_cols = ["model_anomaly"]
    groupby_cols = ["subject_id"]

    # Test no anomalies detected
    with_anomalies["anomaly"] = 1
    with_anomalies["model_anomaly"] = 1

    distance = apm.distance_real_to_detected_anomaly(
        data=with_anomalies,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )
    print(distance)

    # Should have 1 entry per anomaly day, no na vals
    assert distance.shape[0] == N_DAYS
    assert np.any(~distance.distance.isnull())

    # Distances should all be 0
    assert distance["distance"].sum() == 0


def test_distance_mid():
    # Fix to 10 days
    N_DAYS = 10
    N_FEATURES = 3
    FEATURES = {
        f"example_feature_{i}": np.arange(N_DAYS) * i
        for i in range(N_FEATURES)
    }
    TEST_DATA = pd.DataFrame(
        {
            "subject_id": ["test_1"] * N_DAYS,
            "study_day": range(N_DAYS),
            **FEATURES,
        }
    )
    with_anomalies = TEST_DATA.copy()

    # First day has anomaly, detected on 10th
    with_anomalies["anomaly"] = 0
    with_anomalies.loc[with_anomalies.study_day == 0, "anomaly"] = 1
    anomaly_detector_cols = ["model_anomaly"]
    with_anomalies["model_anomaly"] = 0
    with_anomalies.loc[with_anomalies.study_day == 9, "model_anomaly"] = 1
    groupby_cols = ["subject_id"]

    distance = apm.distance_real_to_detected_anomaly(
        data=with_anomalies,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )
    print(distance)

    # Should have 1 entry per anomaly day, no na vals
    assert distance.shape[0] == 1
    assert np.any(~distance.distance.isnull())

    # Distances should all be 0
    assert distance["distance"].sum() == 9


def test_distance_one_detected():
    # Fix to 10 days
    N_DAYS = 10
    N_FEATURES = 3
    FEATURES = {
        f"example_feature_{i}": np.arange(N_DAYS) * i
        for i in range(N_FEATURES)
    }
    TEST_DATA = pd.DataFrame(
        {
            "subject_id": ["test_1"] * N_DAYS,
            "study_day": range(N_DAYS),
            **FEATURES,
        }
    )
    with_anomalies = TEST_DATA.copy()

    # All days but first have anomalies, detected on 10th
    with_anomalies["anomaly"] = 1
    with_anomalies.loc[with_anomalies.study_day == 0, "anomaly"] = 0
    anomaly_detector_cols = ["model_anomaly"]
    with_anomalies["model_anomaly"] = 0
    with_anomalies.loc[with_anomalies.study_day == 9, "model_anomaly"] = 1
    groupby_cols = ["subject_id"]

    distance = apm.distance_real_to_detected_anomaly(
        data=with_anomalies,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )
    print(distance)

    # Should have 1 entry per anomaly day, no na vals
    assert distance.shape[0] == N_DAYS - 1
    assert np.any(~distance.distance.isnull())

    assert distance["distance"].sum() == np.sum(list(range(N_DAYS - 1)))


def test_perfect_correlation():
    n_subjects = 3
    datasets = []
    n_anomalies = [0, 1, 2]
    for i in range(n_subjects):
        for n in n_anomalies:
            subject_data = pd.DataFrame(
                {
                    "subject_id": [f"test_{i}"] * N_DAYS,
                    "study_day": range(N_DAYS),
                }
            )
            subject_data["anomaly"] = 0
            subject_data["model_anomaly"] = 0
            subject_data["setting"] = n

            subject_data.loc[
                subject_data.study_day < n, ["anomaly", "model_anomaly"]
            ] = 1
            datasets.append(subject_data)
    data = pd.concat(datasets)
    data["study"] = "test_study"

    correlation_df = apm.correlateDetectedToInduced(
        data=data,
        anomaly_detector_cols=["model_anomaly"],
        groupby_cols=["study", "subject_id", "setting"],
        corr_across=["study"],
    )
    print(correlation_df)
    assert correlation_df.shape[0] == 1
    assert np.all(correlation_df.rho == 1)
