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

EXPERIMENT = 'exp01'
USE_CACHE = True
PARALLEL = True

# Dataset parameters
N_SUBJECTS = 100
DAYS_OF_DATA = 90
FREQUENCIES = [2, 7, 14, 28]
WINDOW_SIZES = [7, 14, 28] # Can likely reduce to 2
N_FEATURES = 5
KEY_DIFFERENCE = 'history_type'

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
    data['anomaly_freq'] = anomaly_frequency
    data['history_type'] = param_name
    data['window_size'] = window_size
    data['anomaly'] = (
        ((data['study_day'] % anomaly_frequency) == 0) &
        (data['study_day'] > 0)
    )
    data['n_features'] = n_features

    # Run Anomaly Detection
    n_components = 5
    features = [*list(feature_params.keys()), *simulator.added_features]
    if len(features) != n_features:
        raise ValueError(f'Num Features {len(features)}, not as expected ({n_features})')
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
            n_components=n_components
        ),
        anomaly_detection.NMFRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components
        ),
        anomaly_detection.SVMRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=0,
            n_components=n_components
        )
    ]
    for detector in detectors:
        # Remove # of components from name
        dname = detector.name.split('_')[0]
        data[f'{dname}_anomaly'] = np.nan
        for sid in data.subject_id.unique():
            subject_data = data.loc[data.subject_id == sid]
            data.loc[data.subject_id == sid, f'{dname}_anomaly'] = detector.labelAnomaly(subject_data)
    return data



if __name__ == '__main__':
    start = time.perf_counter()

    # DATA SIMULATION
    print('Generating simulated data and running anomaly detection...')


    # File name for the simulated dataset with anomaly detection run
    fname = f'{EXPERIMENT}_nSubjects-{N_SUBJECTS}_nDays-{DAYS_OF_DATA}.csv'
    fpath = Path('cache', fname)

    # If data is cached, do not run anomaly detection only results generation
    if USE_CACHE and fpath.exists():
        print('\tUsing cached data from: ', fpath)
        data_df = pd.read_csv(fpath)
    else:
        datasets = []
        run_list = []

        # Consolidate list of all permutations of simulated data parameters
        for anomaly_frequency in FREQUENCIES:
            for window_size in WINDOW_SIZES:
                feature_param_dict = {
                    'history_all_28': {
                        f'history-{28}_anomalyFrequency-{anomaly_frequency}_{i}': {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": 28,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3
                        } for i in range(N_FEATURES)
                    },
                    'history_0_to_28': {
                        f'history-{feature_history}_anomalyFrequency-{anomaly_frequency}': {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": feature_history,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3
                        } for feature_history in [0, 2, 7, 14, 28]
                    },
                }
                for param_i, (param_name, feature_params) in enumerate(feature_param_dict.items()):
                    run_parameters = {}
                    run_parameters['anomaly_frequency'] = anomaly_frequency
                    run_parameters['feature_params'] = feature_params
                    run_parameters['param_name'] = param_name
                    run_parameters['window_size'] = window_size
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
                if i < 2:
                    continue
                datasets.append(
                    run_ad_on_simulated(**run_params)
                )

        data_df = pd.concat(datasets)
        data_df.to_csv(fpath, index=False)

    anomaly_detector_cols = [d for d in data_df.columns if d.endswith("_anomaly")]
    groupby_cols = ['subject_id', KEY_DIFFERENCE, 'window_size', 'n_features', 'anomaly_freq']
    print(f'Comparing across {KEY_DIFFERENCE}: ', data_df[KEY_DIFFERENCE].unique())

    # PERFORMANCE CALCULATIONS
    print('Calculating Metrics...')

    # Calculate correlation of # anomalies model detects to # induced
    print('\tSpearman R - detected vs induced anomalies')
    correlation_df = anomaly_detection\
        .correlate_detected_to_induced(
            data=data_df,
            anomaly_detector_cols=anomaly_detector_cols,
            groupby_cols=groupby_cols,
            corr_across=[KEY_DIFFERENCE, 'window_size'],
        )


    # Calculate # of day difference between anomaly induced and closest detected anomaly
    print('\tDistance of anomalies to detected anomaly')
    anomaly_detector_behavior = anomaly_detection\
        .distance_real_to_detected_anomaly(
            data=data_df,
            anomaly_detector_cols=anomaly_detector_cols,
            groupby_cols=groupby_cols
        )
    # Calculate accuracy, sensitivity, specificity
    print('\tAccuracy, sensitivity, specificity')
    performance_df = anomaly_detection.performance_metrics(
        data=data_df,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )

    # PLOTTING
    print('Plotting...')
    out_dir = Path('output', EXPERIMENT)
    if not out_dir.exists():
        out_dir.mkdir()

    # Plot correlation of # detected anomalies per subject/anomaly frequency
    hm_size = (10, 7)
    fig, ax = plt.subplots(figsize=hm_size)
    sns.heatmap(
        correlation_df.melt(
            id_vars=[KEY_DIFFERENCE, 'window_size'],
            var_name='model',
            value_name='spearmanr'
        ).pivot_table(
            columns='window_size',
            index=['model', KEY_DIFFERENCE],
            values='spearmanr',
        ),
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        cmap='coolwarm',
        ax=ax
    )
    fname = Path(out_dir, f'spearmanr_heatmap_n{N_SUBJECTS}.png')
    fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
    plt.tight_layout()
    plt.gcf().savefig(str(fname))
    plt.close()

    # Plot performance metrics per condition
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        fig, ax = plt.subplots(figsize=hm_size)
        sns.heatmap(
            performance_df.pivot_table(
                values=metric,
                columns=['anomaly_freq', 'window_size'],
                index=['model', KEY_DIFFERENCE],
            ).round(2),
            annot=True,
            square=True,
            vmin=0,
            vmax=1,
            ax=ax
        )
        fname = Path(out_dir, f'{metric}_heatmap_n{N_SUBJECTS}.png')
        fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    # Plot mean/median distance per condition
    for metric in ['mean', 'median']:
        fig, ax = plt.subplots(figsize=hm_size)
        sns.heatmap(
            anomaly_detector_behavior.pivot_table(
                values='distance',
                columns=['anomaly_freq', 'window_size'],
                index=['model', KEY_DIFFERENCE],
                aggfunc=metric,
            ),
            annot=True,
            square=True,
            vmin=0,
            ax=ax
        )
        fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
        fname = Path(out_dir, f'distance_{metric}_heatmap_n{N_SUBJECTS}.png')
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    print(performance_df.groupby('model').accuracy.describe().round(2))

    stop = time.perf_counter()
    print(f"\nCompleted in {stop - start:0.2f} seconds")