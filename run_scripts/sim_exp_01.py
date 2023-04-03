"""_summary_ Experiment 01: How do 3 simple rolling window based anomaly
detectors function on a 5 feature simulated dataset.
_author_ Samir Akre <sakre@g.ucla.edu>

Simulation induces anomalies at different frequencies (weekly, biweekly, etc.)
and detectors are ran looking different window sizes (weekly, biweekly, etc.)
to see how detected anomalies compare to induced anomalies.

4 Anomaly Detectors, all rolling window based
- Rolling Mean: mean value per feature over window size
- PCA: 3 component PCA trained on window size
- NMF: Non-negative Matrix Factorization trained on window size
- SVM: Support vector machine, based anomaly detection
    - looking at 3 different kernels (radial basis function,
      poly, and sigmoid) 

Output: heatmap of correlation of frequency induced anomalies to detected
anomalies. heatmap of average (mean/median) distance from true anomalies to 
detected anomalies.
"""

import time
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
from p_tqdm import p_map
from pathlib import Path
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
from mhealth_anomaly_detection import simulate_daily
from mhealth_anomaly_detection import anomaly_detection
from mhealth_anomaly_detection import format_axis as fa
from mhealth_anomaly_detection.wrapper_functions import calcSimMetrics

DEBUG = False

EXPERIMENT = "exp01"
USE_CACHE = True
PARALLEL = True
NUM_CPUS = 10

# Dataset parameters
N_SUBJECTS = 100
DAYS_OF_DATA = 120
FREQUENCIES = [2, 7, 14, 28]
WINDOW_SIZES = [7, 14, 28]
N_FEATURES = 5
KEY_DIFFERENCE = "history_type"

if DEBUG:
    N_SUBJECTS = 2
    FREQUENCIES = [2, 7]
    WINDOW_SIZES = [28]
    USE_CACHE = False
    PARALLEL = False

# Ignore divide by 0 error -> expected and happens in PCA
np.seterr(divide="ignore", invalid="ignore")


def run_ad_on_simulated(
    feature_params: Dict,
    param_name: str,
    anomaly_frequency: int,
    window_size: int,
) -> pd.DataFrame:
    # Simulate data according to params above
    simulator = simulate_daily.RandomAnomalySimulator(
        feature_params=feature_params,
        n_days=DAYS_OF_DATA,
        cache_simulation=False,
        n_subjects=N_SUBJECTS,
        sim_type=param_name,
    )

    n_features = N_FEATURES
    # Simulate Data
    data = simulator.simulateData(use_cache=False)
    data["anomaly_freq"] = anomaly_frequency
    data["history_type"] = param_name
    data["window_size"] = window_size
    data["anomaly"] = ((data["study_day"] % anomaly_frequency) == 0) & (
        data["study_day"] > 0
    )
    data["n_features"] = n_features

    # Run Anomaly Detection
    n_components = 3
    features = [*list(feature_params.keys()), *simulator.added_features]
    if len(features) != n_features:
        raise ValueError(
            f"Num Features {len(features)}, not as expected ({n_features})"
        )
    detectors = [
        anomaly_detection.BaseRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
        ),
        anomaly_detection.PCARollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components,
        ),
        anomaly_detection.NMFRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components,
        ),
        anomaly_detection.SVMRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components,
            kernel="poly",
        ),
        anomaly_detection.SVMRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components,
            kernel="sigmoid",
        ),
        anomaly_detection.SVMRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components,
            kernel="rbf",
        ),
        anomaly_detection.IFRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
        ),
    ]
    if DEBUG:
        detectors = detectors[:2]

    def detectAnomalies(grouped) -> pd.DataFrame:
        index_cols = ["subject_id", "study_day", "window_size"]
        _, subject_data = grouped
        for detector in detectors:
            dname = detector.name
            subject_data[f"{dname}_anomaly"] = np.nan
            subject_data[f"{dname}_anomaly_score"] = detector.getContinuous(
                subject_data, recalc_re=True
            )
            subject_data[f"{dname}_anomaly"] = detector.labelAnomaly(
                subject_data, recalc_re=False
            )
        return subject_data.set_index(index_cols)

    anomalies_detected_list = []
    if PARALLEL:
        ad = pd.concat(
            p_map(
                detectAnomalies,
                data.groupby("subject_id"),
                num_cpus=NUM_CPUS,
            )
        )
    else:
        ad = []
        for s in data.groupby("subject_id"):
            print(s[0])
            ad.append(detectAnomalies(s))
        ad = pd.concat(ad)
    anomalies_detected_list.append(ad)

    return pd.concat(anomalies_detected_list).reset_index()


if __name__ == "__main__":
    start = time.perf_counter()

    # DATA SIMULATION
    print("Generating simulated data and running anomaly detection...")

    # File name for the simulated dataset with anomaly detection run
    fname = f"{EXPERIMENT}_nSubjects-{N_SUBJECTS}_nDays-{DAYS_OF_DATA}.csv"
    fpath = Path("cache", fname)

    # If data is cached, do not run anomaly detection only results generation
    if USE_CACHE and fpath.exists():
        print("\tUsing cached data from: ", fpath)
        data_df = pd.read_csv(fpath)
    else:
        datasets = []
        run_list = []

        # Consolidate list of all permutations of simulated data parameters
        for anomaly_frequency in FREQUENCIES:
            for window_size in WINDOW_SIZES:
                feature_param_dict = {
                    "history_all_28": {
                        f"history-{28}_anomalyFrequency-{anomaly_frequency}_{i}": {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": 28,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3,
                        }
                        for i in range(N_FEATURES)
                    },
                    "history_0_to_28": {
                        f"history-{feature_history}_anomalyFrequency-{anomaly_frequency}": {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": feature_history,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3,
                        }
                        for feature_history in [0, 2, 7, 14, 28]
                    },
                }
                for param_i, (param_name, feature_params) in enumerate(
                    feature_param_dict.items()
                ):
                    run_parameters = {}
                    run_parameters["anomaly_frequency"] = anomaly_frequency
                    run_parameters["feature_params"] = feature_params
                    run_parameters["param_name"] = param_name
                    run_parameters["window_size"] = window_size
                    run_list.append(run_parameters)

        def expand_args_run(kwargs):
            return run_ad_on_simulated(**kwargs)

        # Don't parallel process
        datasets = []
        for i, run_params in tqdm(enumerate(run_list)):
            print(f"Running {i+1} of {len(run_list)}")
            datasets.append(run_ad_on_simulated(**run_params))

        data_df = pd.concat(datasets)
        data_df.to_csv(fpath, index=False)

    groupby_cols = [
        "subject_id",
        KEY_DIFFERENCE,
        "window_size",
        "n_features",
        "anomaly_freq",
    ]

    corr, corr_table, performance_cont_df, performance_df = calcSimMetrics(
        data_df, key_difference=KEY_DIFFERENCE, groupby_cols=groupby_cols
    )
    # PLOTTING
    print("Plotting...")
    out_dir = Path("output", EXPERIMENT)
    if not out_dir.exists():
        out_dir.mkdir()

    # Plot correlation of # detected anomalies per subject/anomaly frequency
    hm_size = (10, 7)
    fig, ax = plt.subplots(figsize=hm_size)
    sns.heatmap(
        corr_table,
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        cmap="coolwarm",
        ax=ax,
    )
    fname = Path(out_dir, f"spearmanr_heatmap_n{N_SUBJECTS}.png")
    fa.despine_thicken_axes(
        ax, heatmap=True, fontsize=12, x_tick_fontsize=12, x_rotation=90
    )
    plt.tight_layout()
    plt.gcf().savefig(str(fname))
    plt.close()

    # Plot performance metrics per condition
    for metric in [
        "accuracy",
        "sensitivity",
        "specificity",
        "F1",
        "average_precision",
    ]:
        fig, ax = plt.subplots(figsize=hm_size)
        use_df = performance_df
        if metric == "average_precision":
            use_df = performance_cont_df
        sns.heatmap(
            use_df.pivot_table(
                values=metric,
                columns=["anomaly_freq", "window_size"],
                index=["model", KEY_DIFFERENCE],
            ).round(2),
            annot=True,
            square=True,
            vmin=0,
            vmax=1,
            ax=ax,
        )
        fname = Path(out_dir, f"{metric}_heatmap_n{N_SUBJECTS}.png")
        fa.despine_thicken_axes(
            ax, heatmap=True, fontsize=12, x_tick_fontsize=10
        )
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    print("Overall F1 by model")
    print(
        performance_cont_df.groupby("model")
        .average_precision.describe()
        .round(2)
    )

    stop = time.perf_counter()
    print(f"\nCompleted in {(stop - start)/60:0.2f} minutes")
