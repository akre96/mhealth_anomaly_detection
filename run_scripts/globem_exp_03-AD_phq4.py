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

DEBUG = False
PARALLEL = False

# Ignore divide by 0 error -> expected and happens in PCA
np.seterr(divide='ignore', invalid='ignore')

# Meta params
NUM_CPUS = 6
MAX_MISSING_DAYS = 2
USE_CACHE = False

# Parameters
EXPERIMENT = 'exp04'
YEAR = 2

MIN_DAYS = 7
# To do: Vary window size, period, components
WINDOW_SIZES = [7, 14, 28]
ANOMALY_PERIODS = [1, 2, 3]
N_PARAMS = 4
N_COMPONENTS = [3, 5, 10, 20]
KERNELS = ['poly', 'rbf', 'sigmoid']

# Debugging
if DEBUG:
    WINDOW_SIZES = [7, 14] 
    ANOMALY_PERIODS = [2]
    N_COMPONENTS = [3]
    KERNELS = ['poly']
    USE_CACHE = False 



if __name__ == '__main__':
    start = time.perf_counter()

    # File name for the simulated dataset with anomaly detection run
    fname = f'GLOBEM-{YEAR}_{EXPERIMENT}.csv'
    fpath = Path('cache', fname)

    # If data is cached, do not run anomaly detection only results generation
    if USE_CACHE and fpath.exists():
        print('\tUsing cached data from: ', fpath)
        phq_anomalies = pd.read_csv(fpath)

    else:
        # Load data
        print("\nLoading GLOBEM year 2 dataset...")
        dataset = datasets.GLOBEM(
            data_path='~/Data/mHealth_external_datasets/GLOBEM',
            year=YEAR,
            sensor_data_types=['sleep', 'steps', 'location', 'call'],
        )
        data = dataset.data
        features = dataset.sensor_cols

        ## Impute data
        print("\n\tImputing dataset")
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
        for window_size, i_param in product(WINDOW_SIZES, range(N_PARAMS)):

            # Initiate Anomaly Detectors
            detectors = []
            base_detector = anomaly_detection.BaseRollingAnomalyDetector(
                    features=features,
                    window_size=window_size,
                    max_missing_days=MAX_MISSING_DAYS,
                )
            if i_param == 0:
                detectors.append(base_detector)

            if N_COMPONENTS[i_param] < window_size:
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
            dnames = [d.name for d in detectors]
            print(f'\twindow_size: {window_size}, {dnames}')

            # Detect anomalies
            def detectAnomalies(grouped) -> pd.DataFrame:
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
                return subject_data

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

        anomalies_detected = pd.concat(anomalies_detected_list)
        phq_anomalies_list = []
        print("\n\tCounting anomalies between phq periods")
        for period in ANOMALY_PERIODS:
            for (n_components, window_size), ad_df in anomalies_detected.groupby(
                ['window_size']
            ):

                phq_anomalies = dataset.get_phq_periods(
                    ad_df,
                    features,
                    period,
                )
                phq_anomalies['n_components'] = n_components
                phq_anomalies['window_size'] = window_size
                phq_anomalies_list.append(phq_anomalies)

        print('\n\tSaving results to', fpath)
        pd.concat(phq_anomalies_list).to_csv(fpath, index=False)

    phq_anomalies = pd.read_csv(fpath)
    # QC
    phq_anomalies_qc = phq_anomalies[phq_anomalies.days >= phq_anomalies.period*6]
    print('\tOnly keeping periods that have at least 6 days per week')
    print(f'\t\t from {phq_anomalies.shape[0]} to {phq_anomalies_qc.shape[0]}')
    parameter_cols = ['n_components', 'window_size', 'period']
    anomaly_detector_cols = [
        d for d in phq_anomalies.columns if d.endswith("_anomaly")
    ]
    phq_anom_melt = phq_anomalies_qc.melt(
        id_vars=['subject_id', 'start', 'phq_change'] + parameter_cols,
        value_vars=anomaly_detector_cols,
        value_name='anomalies',
        var_name='detector'
    )

    out_dir = Path('output', EXPERIMENT)
    print(f'\nPlotting to {out_dir}...')
    if not out_dir.exists():
        out_dir.mkdir()
    
    # Plot influence of window size on anomalies changed
    info_cols = [
        'subject_id',
        'period',
        'window_size',
        'n_components',
    ]
    corr_dict = {
        'detector': [],
        'rho': [],
        'p': [],
        'n': [],
        **{
            inf: [] for inf in info_cols
        }
    }
    for info, i_df in phq_anomalies_qc.groupby(info_cols):
        for d in anomaly_detector_cols:
            n = i_df[[d, 'phq_change']].dropna().shape[0]
            rho, p = stats.spearmanr(
                i_df.dropna()[d],
                i_df.dropna()['phq_change']
            )
            if not np.isnan(p):
                corr_dict['detector'].append(d)
                for i in range(len(info_cols)):
                    corr_dict[info_cols[i]].append(info[i])
                corr_dict['n'].append(n)
                corr_dict['rho'].append(rho)
                corr_dict['p'].append(p)
    corr = pd.DataFrame(corr_dict) 

    corr_table = corr.pivot_table(
        index=['detector', 'window_size'],
        columns=['period', 'n_components'],
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
    fname = Path(out_dir, 'spearmanr_heatmap.png')
    fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
    plt.tight_layout()
    plt.gcf().savefig(str(fname))
    plt.close()

    stop = time.perf_counter()
    print(f"\nCompleted in {stop - start:0.2f} seconds")

