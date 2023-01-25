
"""_summary_ Experiment 02: Extension of exp01, now comparing how number of features affects performance
_author_ Samir Akre <sakre@g.ucla.edu>


Output: heatmap of correlation of frequency induced anomalies to detected
anomalies. heatmap of average (mean/median) distance from true anomalies to
detected anomalies. Heatmap of sensitivity/specificity/accuracy of models at
predicting anomalous days
"""
import time
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from typing import Dict
import seaborn as sns
import matplotlib.pyplot as plt
from mhealth_anomaly_detection import simulate_daily
from mhealth_anomaly_detection import anomaly_detection
from mhealth_anomaly_detection import format_axis as fa

EXPERIMENT = 'exp02'
USE_CACHE = False 

# Dataset parameters
N_SUBJECTS = 100
DAYS_OF_DATA = 90
FREQUENCIES = [2, 7, 14, 28]
WINDOW_SIZES = [7, 14, 28] # Can likely reduce to 2
N_FEATURES_LIST = [5, 10, 25, 100]

# Test it run
FREQUENCIES = [14, 28]
WINDOW_SIZES = [14, 28] # Can likely reduce to 2
N_FEATURES_LIST = [5, 10]

def run_ad_on_simulated(
    feature_params: Dict,
    param_name: str,
    anomaly_frequency: int,
    window_size: int,
    n_features: int,
) -> pd.DataFrame:

    # Simulate data according to params above
    simulator = simulate_daily.RandomAnomalySimulator(
        feature_params=feature_params,
        n_days=DAYS_OF_DATA,
        cache_simulation=False,
        n_subjects=N_SUBJECTS,
        sim_type=param_name,
    )

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
    n_components = round(n_features/2)
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
            for n_features in N_FEATURES_LIST:
                for window_size in WINDOW_SIZES:
                    run_parameters = {}
                    run_parameters['anomaly_frequency'] = anomaly_frequency
                    run_parameters['n_features'] = n_features
                    run_parameters['feature_params'] = {
                        f'history-{28}_anomalyFrequency-{anomaly_frequency}_{i}': {
                            "min": 0,
                            "max": 10,
                            "mean": 5,
                            "std": 2,
                            "history_len": 28,
                            "anomaly_frequency": anomaly_frequency,
                            "anomaly_std_scale": 3
                        } for i in range(n_features)
                    }
                    run_parameters['param_name'] = 'history_all_28'
                    run_parameters['window_size'] = window_size
                    run_list.append(run_parameters)

        # Run parameters
        for run_params in tqdm(run_list):
            print('post', run_params['n_features'])
            datasets.append(
                run_ad_on_simulated(**run_params)
            )

        data_df = pd.concat(datasets)
        data_df.to_csv(fpath, index=False)

    anomaly_detector_cols = [d for d in data_df.columns if d.endswith("_anomaly")]
    groupby_cols = ['subject_id', 'anomaly_freq', 'window_size', 'n_features']
    print(data_df['n_features'].unique())

    # PERFORMANCE CALCULATIONS
    print('Calculating Metrics...')

    # Calculate correlation of # anomalies model detects to # induced
    print('\tSpearman R - detected vs induced anomalies')
    correlation_df = anomaly_detection\
        .correlate_detected_to_induced(
            data=data_df,
            anomaly_detector_cols=anomaly_detector_cols,
            groupby_cols=groupby_cols,
            corr_across=['n_features', 'window_size'],
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

    # Plot correlation of # detected anomalies per subject/anomaly frequency
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        correlation_df.melt(
            id_vars=['n_features', 'window_size'],
            var_name='model',
            value_name='spearmanr'
        ).pivot_table(
            columns='window_size',
            index=['n_features', 'model'],
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
    fname = Path('output', EXPERIMENT, f'spearmanr_heatmap_n{N_SUBJECTS}.png')
    fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
    plt.tight_layout()
    plt.gcf().savefig(str(fname))
    plt.close()

    # Plot performance metrics per condition
    for metric in ['accuracy', 'sensitivity', 'specificity']:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            performance_df.pivot_table(
                values=metric,
                columns=['anomaly_freq', 'window_size'],
                index=['n_features', 'model'],
            ).round(2),
            annot=True,
            square=True,
            vmin=0,
            vmax=1,
            ax=ax
        )
        fname = Path('output', EXPERIMENT, f'{metric}_heatmap_n{N_SUBJECTS}.png')
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
                columns=['anomaly_freq', 'window_size'],
                index=['n_features', 'model'],
                aggfunc=metric,
            ),
            annot=True,
            square=True,
            vmin=0,
            ax=ax
        )
        fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
        fname = Path('output', EXPERIMENT, f'distance_{metric}_heatmap_n{N_SUBJECTS}.png')
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

    print(performance_df.groupby('model').accuracy.describe().round(2))

    # TODO: calculate how many induced anomalies were missed [no detected anomaly before next anomaly]
    # TODO: calculate how many detected anomalies were before the first induced

    stop = time.perf_counter()
    print(f"\nCompleted in {stop - start:0.2f} seconds")