"""_summary_ Experiment 03: Extension of exp01, now comparing how correlated features affect performance
_author_ Samir Akre <sakre@g.ucla.edu>


Output: heatmap of correlation of frequency induced anomalies to detected
anomalies. heatmap of average (mean/median) distance from true anomalies to
detected anomalies. Heatmap of sensitivity/specificity/accuracy of models at
predicting anomalous days
"""
import sys

# Make imports work
# TODO: Remove this dependency -- worked fine when using poetry, but not just python3
sys.path.insert(0, "/Users/sakre/Code/dgc/mhealth_anomaly_detection")
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

EXPERIMENT = "exp03"
USE_CACHE = False
PARALLEL = False

# Dataset parameters
N_SUBJECTS = 100
DAYS_OF_DATA = 120
FREQUENCIES = [28]
WINDOW_SIZES = [14]
N_FEATURES = 24
N_CORR = 2
CORRELATION_LIST = ["Independent", "Linear", "Non-linear"]
KEY_DIFFERENCE = "correlation_type"


def run_ad_on_simulated(
    feature_params: Dict,
    param_name: str,
    anomaly_frequency: int,
    window_size: int,
    correlation_type: str,
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
    if correlation_type == "Linear":
        data = simulator.addCorrelatedFeatures(
            data=data, n_feats=N_CORR, noise_scale=0.1
        )
    if correlation_type == "Non-linear":
        data = simulator.addReLuFeatures(
            data=data,
            n_feats=N_CORR,
        )

    data["correlation_type"] = correlation_type
    data["n_corr"] = N_CORR
    data["anomaly_freq"] = anomaly_frequency
    data["history_type"] = param_name
    data["window_size"] = window_size
    data["anomaly"] = ((data["study_day"] % anomaly_frequency) == 0) & (
        data["study_day"] > 0
    )
    data["n_features"] = n_features

    # Run Anomaly Detection
    n_components = 5
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
        anomaly_detection.PCAGridRollingAnomalyDetector(
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
    ]
    for detector in detectors:
        # Remove # of components from name
        dname = detector.name
        data[f"{dname}_anomaly"] = np.nan
        for sid in data.subject_id.unique():
            subject_data = data.loc[data.subject_id == sid]
            data.loc[
                data.subject_id == sid, f"{dname}_anomaly_score"
            ] = detector.getContinuous(subject_data, recalc_re=True)

            data.loc[
                data.subject_id == sid, f"{dname}_anomaly"
            ] = detector.labelAnomaly(subject_data, recalc_re=False)
    return data


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
            for correlation_type in CORRELATION_LIST:
                for window_size in WINDOW_SIZES:
                    run_parameters = {}
                    run_parameters["anomaly_frequency"] = anomaly_frequency
                    run_parameters["correlation_type"] = correlation_type
                    fp_feats = N_FEATURES
                    if correlation_type != "Independent":
                        fp_feats = int(N_FEATURES / (N_CORR + 1))
                    run_parameters["feature_params"] = {
                        f"history-{28}_anomalyFrequency-{anomaly_frequency}_{i}": {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": 28,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3,
                        }
                        for i in range(fp_feats)
                    }
                    run_parameters["param_name"] = "history_all_28"
                    run_parameters["window_size"] = window_size
                    run_list.append(run_parameters)

        def expand_args_run(kwargs):
            return run_ad_on_simulated(**kwargs)

        # Run parameters - simulation + anomaly detection
        if PARALLEL:
            # Parallel process
            datasets = p_map(expand_args_run, run_list)
        else:
            # Don't parallel process
            datasets = []
            for i, run_params in tqdm(enumerate(run_list)):
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

    # PERFORMANCE CALCULATIONS
    print("Calculating Metrics...")

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
    for metric in ["accuracy", "sensitivity", "specificity", "F1"]:
        fig, ax = plt.subplots(figsize=hm_size)
        sns.heatmap(
            performance_df.pivot_table(
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

    print(
        performance_cont_df.groupby("model")
        .average_precision.describe()
        .round(2)
    )

    stop = time.perf_counter()
    print(f"\nCompleted in {(stop - start)/60:0.2f} minutes")
