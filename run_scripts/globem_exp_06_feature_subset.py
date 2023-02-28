"""_summary_ Experiment 05: Are curated featuers better for AD?
_author_ Samir Akre <sakre@g.ucla.edu>

This work looks at the correlation between detected anomalies and change in 
PHQ-4 score on the year 2 GLOBEM dataset. This looks at 192 subjects daily
sleep data exclusively.

Output: heatmap of correlation of PHQ-4 change to detected
anomalies. 
"""
import time
import pandas as pd
import numpy as np
import scipy.stats as stats
from p_tqdm import p_map
from pathlib import Path
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa
# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

from mhealth_anomaly_detection import datasets
from mhealth_anomaly_detection import impute
from mhealth_anomaly_detection import anomaly_detection
from mhealth_anomaly_detection import format_axis as fa

# Runtime parameters
DEBUG = False

PARALLEL = True
USE_CACHE = True
USE_CACHE_INTERMEDIATE = False

# Ignore divide by 0 error -> expected and happens in PCA
np.seterr(divide='ignore', invalid='ignore')

# Meta params
NUM_CPUS = 6
MAX_MISSING_DAYS = 2
EXPERIMENT = 'exp06'

# Dataset Parameters
YEAR = 2
SENSOR_TYPES = ['sleep', 'steps', 'location', 'call']
FEATURES = [
    'f_loc:phone_locations_doryab_locationentropy:allday',
    'f_loc:phone_locations_barnett_circdnrtn:allday',
    'f_steps:fitbit_steps_intraday_rapids_sumsteps:allday',
    'f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:allday',
    'f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday',
    'f_slp:fitbit_sleep_intraday_rapids_countepisodeasleepunifiedmain:allday',
    'f_slp:fitbit_sleep_summary_rapids_firstbedtimemain:allday',
    'f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:allday',
    'f_call:phone_calls_rapids_missed_count:allday',
    'f_call:phone_calls_rapids_incoming_count:allday',
    'f_call:phone_calls_rapids_outgoing_count:allday',
    'f_call:phone_calls_rapids_outgoing_sumduration:allday',
    'sleep_missing',
    'steps_missing',
    'location_missing',
    'call_missing',
]
MIN_DAYS = 7

# Detector parameters
WINDOW_SIZES = [7, 14, 28]
ANOMALY_PERIODS = [1, 2, 3]
N_COMPONENTS = [3, 5, 10]
KERNELS = ['poly', 'rbf', 'sigmoid']

# Debugging
if DEBUG:
    WINDOW_SIZES = [7, 14] 
    ANOMALY_PERIODS = [2]
    N_COMPONENTS = [3]
    KERNELS = ['poly']
    USE_CACHE = False 
    USE_CACHE_INTERMEDIATE = False

N_PARAMS = max([len(N_COMPONENTS), len(KERNELS)])

if __name__ == '__main__':
    start = time.perf_counter()

    # File name for the simulated dataset with anomaly detection run
    folder = Path('cache')
    if DEBUG:
        folder = Path('cache', 'debug')
        if not folder.exists():
            folder.mkdir()

    fname = f'GLOBEM-{YEAR}_{EXPERIMENT}.csv'
    fpath = Path(folder, fname)

    inter_fname = f'GLOBEM-{YEAR}_{EXPERIMENT}_intermediate.csv'
    inter_fpath = Path(folder, inter_fname)

    # If data is cached, do not run anomaly detection only results generation
    if USE_CACHE and fpath.exists():
        print('\tUsing cached data from: ', fpath)
        phq_anomalies = pd.read_csv(fpath)

    else:
        print("\nLoading GLOBEM year 2 dataset...")
        dataset = datasets.GLOBEM(
            data_path='~/Data/mHealth_external_datasets/GLOBEM',
            year=YEAR,
            sensor_data_types=SENSOR_TYPES,
        )
        data = dataset.data
        if DEBUG:
            use_ids = data.subject_id.unique()[:10]
            data = data[data.subject_id.isin(use_ids)]
        features = FEATURES
        # Load data
        if not (USE_CACHE_INTERMEDIATE and inter_fpath.exists()):
            ## Impute data
            print("\nImputing dataset")
            imputer = IterativeImputer(
                initial_strategy='median',
                keep_empty_features=True,
                skip_complete=True
            )

            imputed = impute.rollingImpute(
                data,
                features,
                MIN_DAYS,
                imputer,
                num_cpus=NUM_CPUS
            )
            anomalies_detected_list = []
            print("\nRunning anomaly detection in different conditions...")
            conditions = list(product(WINDOW_SIZES, range(N_PARAMS)))
            for i, (window_size, i_param) in enumerate(conditions):
                # Initiate Anomaly Detectors
                detectors = []
                base_detector = anomaly_detection.BaseRollingAnomalyDetector(
                        features=features,
                        window_size=window_size,
                        max_missing_days=MAX_MISSING_DAYS,
                    )
                if i_param == 0:
                    detectors.append(base_detector)

                if i_param < len(N_COMPONENTS) and (N_COMPONENTS[i_param] < window_size):
                    detectors.append(
                        anomaly_detection.PCARollingAnomalyDetector(
                            features=features,
                            window_size=window_size,
                            max_missing_days=MAX_MISSING_DAYS,
                            n_components=N_COMPONENTS[i_param]
                        )
                    )
                    detectors.append(
                        anomaly_detection.NMFRollingAnomalyDetector(
                            features=features,
                            window_size=window_size,
                            max_missing_days=MAX_MISSING_DAYS,
                            n_components=N_COMPONENTS[i_param]
                        )
                    )
                if i_param < len(KERNELS):
                    detectors.append(
                        anomaly_detection.SVMRollingAnomalyDetector(
                            features=features,
                            window_size=window_size,
                            max_missing_days=MAX_MISSING_DAYS,
                            kernel=KERNELS[i_param],
                            n_components=5
                        )
                    )
                if len(detectors) == 0:
                    continue
                dnames = [d.name for d in detectors]
                print(f'\t {i+1} of {len(conditions)}: window_size: {window_size}, {dnames}')

                # Detect anomalies
                def detectAnomalies(grouped) -> pd.DataFrame:
                    index_cols = [
                        'subject_id',
                        'study_day',
                        'window_size'
                    ]
                    _, subject_data = grouped
                    for detector in detectors:
                        dname = detector.name
                        if i_param > 0:
                            if dname in ['RollingMean']:
                                continue
                        subject_data[f'{dname}_anomaly'] = np.nan
                        subject_data[f'{dname}_anomaly'] = detector\
                            .labelAnomaly(subject_data)
                    subject_data['window_size'] = window_size
                    return subject_data.set_index(index_cols)

                if PARALLEL:
                    ad = pd.concat(
                        p_map(
                            detectAnomalies,
                            imputed.groupby('subject_id'),
                            num_cpus=NUM_CPUS
                        )
                    )
                else:
                    ad = []
                    for s in imputed.groupby('subject_id'):
                        print(s[0])
                        ad.append(detectAnomalies(s))
                    ad = pd.concat(ad)
                anomalies_detected_list.append(ad)

            anomalies_detected = pd.concat(anomalies_detected_list).reset_index()
            anomalies_detected.to_csv(inter_fpath, index=False)

        anomalies_detected = pd.read_csv(inter_fpath)
        phq_anomalies_list = []
        print("\n\tCounting anomalies between phq periods")
        for period in ANOMALY_PERIODS:
            for (window_size), ad_df in anomalies_detected.groupby(
                ['window_size']
            ):

                phq_anomalies = dataset.get_phq_periods(
                    ad_df,
                    features,
                    period,
                )
                phq_anomalies['window_size'] = window_size
                phq_anomalies_list.append(phq_anomalies)

        print('\n\tSaving results to', fpath)
        pd.concat(phq_anomalies_list).to_csv(fpath, index=False)

    phq_anomalies = pd.read_csv(fpath)
    # QC
    phq_anomalies_qc = phq_anomalies[phq_anomalies.days >= phq_anomalies.period*6]
    print('\tOnly keeping periods that have at least 6 days per week')
    print(f'\t\t from {phq_anomalies.shape[0]} to {phq_anomalies_qc.shape[0]}')
    parameter_cols = ['window_size', 'period']
    anomaly_detector_cols = [
        d for d in phq_anomalies.columns if d.endswith("_anomaly")
    ]
    phq_anom_melt = phq_anomalies_qc.melt(
        id_vars=['subject_id', 'start', 'phq_change', 'phq_stop', 'phq_start'] + parameter_cols,
        value_vars=anomaly_detector_cols,
        value_name='anomalies',
        var_name='detector'
    )

    out_dir = Path('output', EXPERIMENT)
    if DEBUG:
        out_dir = Path('output', 'debug', EXPERIMENT)
    print(f'\nPlotting to {out_dir}...')
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
    
    # Plot influence of window size on anomalies changed
    info_cols = [
        'period',
        'window_size',
    ]
    for target in ['phq_change', 'phq_stop', 'phq_start']:
        corr = anomaly_detection.correlateDetectedToOutcome(
            phq_anomalies_qc,
            anomaly_detector_cols,
            outcome_col=target,
            groupby_cols=info_cols,
        )
        corr['r2'] = corr['rho'] ** 2
        for metric in ['rho', 'r2']:
            corr_table = corr.pivot_table(
                index=['detector'],
                columns=['window_size', 'period'],
                values='rho',
                aggfunc='median'
            )
            hm_size = (10, 7)
            fig, ax = plt.subplots(figsize=hm_size)
            sns.heatmap(
                corr_table,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                annot=True,
                cmap='coolwarm',
                ax=ax
            )
            fname = Path(out_dir, f'spearmanr_{metric}_{target}_heatmap.png')
            fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
            plt.tight_layout()
            plt.gcf().savefig(str(fname))
            plt.close()


    stop = time.perf_counter()
    print(f"\nCompleted in {stop - start:0.2f} seconds")

