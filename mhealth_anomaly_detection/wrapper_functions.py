""" Wrapper functions for commonly performed tasks
"""
import pandas as pd
from mhealth_anomaly_detection import anomaly_performance_metrics


def calcSimMetrics(data_df: pd.DataFrame, key_difference, groupby_cols):
    anomaly_detector_cols = [
        d for d in data_df.columns if d.endswith("_anomaly")
    ]
    anomaly_detector_cont_cols = [
        d for d in data_df.columns if d.endswith("_anomaly_score")
    ]
    print(
        f"Comparing across {key_difference}: ",
        data_df[key_difference].unique(),
    )

    # PERFORMANCE CALCULATIONS
    print("Calculating Metrics...")

    # Calculate correlation of # anomalies model detects to # induced
    print("\tSpearman R - detected vs induced anomalies")
    corr = anomaly_performance_metrics.correlateDetectedToInduced(
        data=data_df,
        anomaly_detector_cols=anomaly_detector_cols,
        groupby_cols=groupby_cols,
        corr_across=[key_difference, "window_size"],
    )
    corr_table = corr.pivot_table(
        index=["detector"],
        columns=["window_size", key_difference],
        values="rho",
        aggfunc="median",
    )

    # Calculate accuracy, sensitivity, specificity
    print("\tAverage precision")
    performance_cont_df = (
        anomaly_performance_metrics.continuousPerformanceMetrics(
            data=data_df,
            groupby_cols=groupby_cols,
            anomaly_detector_cols=anomaly_detector_cont_cols,
        )
    )

    # Calculate accuracy, sensitivity, specificity
    print("\tAccuracy, sensitivity, specificity")
    performance_df = anomaly_performance_metrics.binaryPerformanceMetrics(
        data=data_df,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )
    return corr, corr_table, performance_cont_df, performance_df
