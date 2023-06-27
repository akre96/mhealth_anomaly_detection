import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from scipy import stats
from typing import List
from tqdm import tqdm


# Calculate accuracy, sensitivity and specificity
def binaryPerformanceMetrics(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str] = [
        "subject_id",
        "history_type",
        "window_size",
        "anomaly_freq",
    ],
) -> pd.DataFrame:
    models = [c.split("_anomaly")[0] for c in anomaly_detector_cols]
    performance_dict = {c: [] for c in groupby_cols}
    performance_dict["model"] = []
    performance_dict["true_positives"] = []
    performance_dict["true_negatives"] = []
    performance_dict["false_positives"] = []
    performance_dict["false_negatives"] = []

    gc = groupby_cols
    if len(groupby_cols) == 1:
        gc = groupby_cols[0]
    for info, subject_data in tqdm(data.groupby(gc)):
        subject_data["anomaly"] = subject_data["anomaly"].astype(bool)
        # Fix error if only one groupby item, info is a string, not a tuple[str]
        if type(info) == str:
            info = [info]
        for model in models:
            subject_data[model + "_anomaly"] = subject_data[
                model + "_anomaly"
            ].astype(bool)
            for i, val in enumerate(info):
                performance_dict[groupby_cols[i]].append(val)
            performance_dict["model"].append(model)
            performance_dict["true_positives"].append(
                (
                    subject_data["anomaly"] & subject_data[model + "_anomaly"]
                ).sum()
            )
            performance_dict["true_negatives"].append(
                (
                    ~subject_data["anomaly"]
                    & ~subject_data[model + "_anomaly"]
                ).sum()
            )
            performance_dict["false_negatives"].append(
                (
                    subject_data["anomaly"] & ~subject_data[model + "_anomaly"]
                ).sum()
            )
            performance_dict["false_positives"].append(
                (
                    ~subject_data["anomaly"] & subject_data[model + "_anomaly"]
                ).sum()
            )

    performance_df = pd.DataFrame(performance_dict)
    performance_df["sensitivity"] = performance_df["true_positives"] / (
        performance_df["true_positives"] + performance_df["false_negatives"]
    )
    performance_df["precision"] = performance_df["true_positives"] / (
        performance_df["true_positives"] + performance_df["false_positives"]
    )
    performance_df["specificity"] = performance_df["true_negatives"] / (
        performance_df["true_negatives"] + performance_df["false_positives"]
    )
    performance_df["accuracy"] = performance_df[
        ["true_positives", "true_negatives"]
    ].sum(axis=1) / performance_df[
        [
            "true_positives",
            "true_negatives",
            "false_positives",
            "false_negatives",
        ]
    ].sum(
        axis=1
    )
    performance_df["F1"] = (
        2
        * performance_df["sensitivity"]
        * performance_df["precision"]
        / performance_df[["sensitivity", "precision"]].sum(axis=1)
    )
    return performance_df


def continuousPerformanceMetrics(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str] = [
        "subject_id",
        "history_type",
        "window_size",
        "anomaly_freq",
    ],
) -> pd.DataFrame:
    models = [c.split("_anomaly_score")[0] for c in anomaly_detector_cols]
    performance_dict = {c: [] for c in groupby_cols}
    performance_dict["model"] = []
    performance_dict["average_precision"] = []

    for info, subject_data in tqdm(data.groupby(groupby_cols)):
        # Fix error if one groupby item: info is a string, not a tuple[str]
        if type(info) == str:
            info = [info]

        for model in models:
            for i, val in enumerate(info):
                performance_dict[groupby_cols[i]].append(val)
            score_col = f"{model}_anomaly_score"
            use_df = subject_data[["anomaly", score_col]].dropna()
            performance_dict["model"].append(model)
            performance_dict["average_precision"].append(
                average_precision_score(
                    use_df["anomaly"],
                    use_df[f"{model}_anomaly_score"],
                )
            )
    performance_df = pd.DataFrame(performance_dict)
    return performance_df


# Find distance of induced anomaly to closest detected anomaly
def distance_real_to_detected_anomaly(
    data: pd.DataFrame,
    groupby_cols: List[str],
    anomaly_detector_cols: List[str],
) -> pd.DataFrame:
    anomaly_detector_distances = []

    models = [c.split("_anomaly")[0] for c in anomaly_detector_cols]
    gc = groupby_cols
    if len(groupby_cols) == 1:
        gc = groupby_cols[0]
    for info, subject_data in tqdm(data.groupby(gc)):
        for model in models:
            subject_data[model + "_anomaly"] = subject_data[
                model + "_anomaly"
            ].astype(bool)
        subject_data["anomaly"] = subject_data["anomaly"].astype(bool)
        # day of detected anomaly
        anomaly_days = {
            c: subject_data.loc[
                subject_data[c + "_anomaly"], "study_day"
            ].values
            for c in models
        }

        # Actual induced anomalies
        real_anomaly = subject_data.loc[
            subject_data["anomaly"], "study_day"
        ].values

        # Initialize with NaN values
        anomaly_distance = {
            c: np.full(real_anomaly.shape, np.nan) for c in models
        }
        # Calculate distance for each induced anomaly to closest future detected anomaly
        for i in range(real_anomaly.shape[0]):
            for c in models:
                distances = anomaly_days[c] - real_anomaly[i]
                pos_distances = distances[distances >= 0]

                # If no future detected anomaly, fill distance with np.nan
                if np.any(pos_distances >= 0):
                    min_pos = np.min(pos_distances)
                else:
                    min_pos = np.nan
                anomaly_distance[c][i] = min_pos

        anomaly_distance = pd.DataFrame(anomaly_distance)
        anomaly_distance["study_day"] = real_anomaly
        for i, c in enumerate(groupby_cols):
            anomaly_distance[c] = info[i]
        anomaly_detector_distances.append(anomaly_distance)

    # for each detector's anomalies, find closest day of real anomaly on or after detected, and calculate distance
    # If no real anomaly before detected anomaly, distance = np.nan
    anomaly_detector_distance_df = pd.concat(anomaly_detector_distances)
    anomaly_detector_distance_df = anomaly_detector_distance_df.melt(
        id_vars=["study_day", *groupby_cols],
        var_name="model",
        value_name="distance",
    )
    return anomaly_detector_distance_df


def correlateDetectedToOutcome(
    detected_anomalies: pd.DataFrame,
    anomaly_detector_cols: List[str],
    outcome_col: str,
    groupby_cols: List[str],
) -> pd.DataFrame:
    corr_dict = {
        "detector": [],
        "rho": [],
        "p": [],
        "n": [],
        **{inf: [] for inf in groupby_cols},
    }
    gc = groupby_cols
    if len(groupby_cols) == 1:
        gc = groupby_cols[0]
    for info, i_df in detected_anomalies.groupby(gc):
        for d in anomaly_detector_cols:
            d_df = i_df[[d, outcome_col]].dropna()
            n = d_df.shape[0]
            if (d_df[d].nunique() == 1) or (d_df[outcome_col].nunique() == 1):
                rho, p = (0, 1)
            else:
                rho, p = stats.spearmanr(d_df[d], d_df[outcome_col])
            corr_dict["detector"].append(d)
            for i in range(len(groupby_cols)):
                corr_dict[groupby_cols[i]].append(info[i])
            corr_dict["n"].append(n)
            corr_dict["rho"].append(rho)
            corr_dict["p"].append(p)
    return pd.DataFrame(corr_dict).rename(
        {c: c.split("_anomaly")[0] for c in anomaly_detector_cols}
    )


# Correlation between # detected and induced anomalies in simulation
def correlateDetectedToInduced(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str],
    corr_across: List[str],
) -> pd.DataFrame:
    for c in corr_across:
        if c not in groupby_cols:
            raise ValueError(
                f"corr_across value {c} not found in groupby_cols"
            )

    # Ensure bool
    for c in ["anomaly", *anomaly_detector_cols]:
        data[c] = data[c].astype(bool)

    # Anomalies detected per subject/model
    detected_anomalies = (
        data.groupby(groupby_cols)[["anomaly"] + anomaly_detector_cols]
        .sum(numeric_only=False)
        .reset_index()
    )

    return correlateDetectedToOutcome(
        detected_anomalies, anomaly_detector_cols, "anomaly", corr_across
    )
