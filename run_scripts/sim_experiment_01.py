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
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from mhealth_anomaly_detection import simulate_daily
from mhealth_anomaly_detection import anomaly_detection
from mhealth_anomaly_detection import format_axis as fa

USE_CACHE = True
#USE_CACHE = False

if __name__ == '__main__':
    start = time.perf_counter()

    # Dataset parameters
    n_subjects = 100
    days_of_data = 90
    frequencies = [7, 14, 28]
    n_features = 5

    # File name for the simulated dataset with anomaly detection run
    fname = f'exp01_nSubjects-{n_subjects}_nDays-{days_of_data}_n_features-{n_features}.csv'
    fpath = Path('cache', fname)

    # If data is cached, do not run anomaly detection only results generation
    if USE_CACHE and fpath.exists():
        print('Using cached data from: ', fpath)
        data_df = pd.read_csv(fpath)
    else:
        datasets = []
        for anomaly_frequency in frequencies:
            print('Anomaly frequency: ', anomaly_frequency, 'of', frequencies)
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
                    } for i in range(n_features)
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
                print('\t Parameter set', param_i+1, 'of', len(feature_param_dict))
                # Simulate data according to params above
                simulator = simulate_daily.RandomAnomalySimulator(
                    feature_params=feature_params,
                    n_days=days_of_data,
                    cache_simulation=False,
                    n_subjects=n_subjects,
                    sim_type=param_name,
                )


                # Anomaly detector window (training) size
                window_sizes = [7, 14, 28]
                for window_i, window_size in enumerate(window_sizes):
                    print('\t\t Window size', window_i+1, 'of', len(window_sizes))

                    # Simulate Data
                    data = simulator.simulateData(use_cache=False)
                    data['anomaly_freq'] = anomaly_frequency
                    data['history_type'] = param_name
                    data['window_size'] = window_size
                    data['anomaly'] = (
                        ((data['study_day'] % anomaly_frequency) == 0) &
                        (data['study_day'] > 0)
                    )

                    # Run Anomaly Detection
                    n_components = 3
                    features = list(feature_params.keys())
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
                        data[f'{detector.name}_anomaly'] = np.nan
                        for sid in data.subject_id.unique():
                            subject_data = data.loc[data.subject_id == sid]
                            data.loc[data.subject_id == sid, f'{detector.name}_anomaly'] = detector.labelAnomaly(subject_data)
                    datasets.append(data)
        data_df = pd.concat(datasets)
        data_df.to_csv(fpath, index=False)

    anomaly_detector_cols = [d for d in data_df.columns if d.endswith("_anomaly")]

    # Anomalies detected per subject/model
    detected_anomalies = data_df.groupby(
        ['subject_id', 'history_type', 'anomaly_freq', 'window_size']
    )[['anomaly'] + anomaly_detector_cols].sum(numeric_only=False)

    split_by = ['history_type', 'window_size']
    
    # Results are for correlation of # anomalies detected to # induced
    results = []
    counts = []
    for c in anomaly_detector_cols:
        # Get spearman correlation of # detected anomalies per subject/anomaly frequency
        # Separate different data generation types (history_type) and model training window size
        res = detected_anomalies.reset_index()\
            .groupby(split_by)[['anomaly', c]]\
            .corr(method='spearman').reset_index()
        res = res[res[f'level_{len(split_by)}'] == 'anomaly'][split_by + [c]]\
            .rename(columns={c:c.split('_anomaly')[0]})\
            .set_index(split_by)
        results.append(res)
    results_df = pd.concat(results, axis=1).reset_index()

    # Plot correlation of # detected anomalies per subject/anomaly frequency
    ax = sns.heatmap(
        results_df.melt(
            id_vars=split_by,
            var_name='model',
            value_name='spearmanr'
        ).pivot_table(
            columns='window_size',
            index=['history_type', 'model'],
            values='spearmanr',
        ),
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        annot=True,
        cmap='coolwarm'
    )
    fname = Path('output', 'exp01', f'spearmanr_heatmap_n{n_subjects}.png')
    plt.tight_layout()
    plt.gcf().savefig(str(fname))
    plt.close()

    # Calculate # of day difference between anomaly induced and closest detected anomaly
    groupby_cols = ['subject_id', 'anomaly_freq'] + split_by
    anomaly_detector_behavior = anomaly_detection\
        .distance_real_to_detected_anomaly(
            data=data_df,
            anomaly_detector_cols=anomaly_detector_cols,
            groupby_cols=groupby_cols
        )
    # Calculate accuracy, sensitivity, specificity
    performance_df = anomaly_detection.performance_metrics(
        data=data_df,
        groupby_cols=groupby_cols,
        anomaly_detector_cols=anomaly_detector_cols,
    )

    # Plot performance metrics per condition
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            performance_df.pivot_table(
                values=metric,
                columns=['window_size', 'anomaly_freq'],
                index=['history_type', 'model'],
            ).round(2),
            annot=True,
            square=True,
            vmin=0,
            vmax=1,
            ax=ax
        )
        fname = Path('output', 'exp01', f'{metric}_heatmap_n{n_subjects}.png')
        fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    # Plot mean/median distance per condition
    for metric in ['mean', 'median']:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            anomaly_detector_behavior.pivot_table(
                values='distance',
                columns=['window_size', 'anomaly_freq'],
                index=['history_type', 'model'],
                aggfunc=metric,
            ),
            annot=True,
            square=True,
            vmin=0,
            ax=ax
        )
        fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
        fname = Path('output', 'exp01', f'distance_{metric}_heatmap_n{n_subjects}.png')
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    print(anomaly_detector_behavior.describe())

    # TODO: calculate how many induced anomalies were missed [no detected anomaly before next anomaly]
    # TODO: calculate how many detected anomalies were before the first induced

    stop = time.perf_counter()
    print(f"\nCompleted in {stop - start:0.2f} seconds")