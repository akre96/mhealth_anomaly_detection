""" Summary: Anomaly detection methods on simple features from healthKit data from the OPTIMA study
Output will be compared to self report scores. In particular to see if important features
relate to known disruptions to behavior.

Author: Samir Akre <sakre@g.ucla.edu>

Output: CSV with anomalies detected in healthKit data from the OPTIMA study
"""
import time
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product
from p_tqdm import p_map
from mhealth_anomaly_detection import datasets
from mhealth_anomaly_detection import impute
from mhealth_anomaly_detection import anomaly_detection
from functools import partial

# now you can import normally from sklearn.impute
from sklearn.impute import SimpleImputer 

# Meta params
NUM_CPUS = 10
MAX_MISSING_DAYS = 6
EXPERIMENT = "OPTIMA_exp02"

# Anomaly detector Parameters
WINDOW_SIZES = [7, 14, 28]
WINDOW_SIZES = [14]
N_COMPONENTS = [3]
MIN_DAYS = 7
MAX_MISSING_DAYS = 2

# Runtime parameters
USE_CACHE = False
PARALLEL = False

def detectAnomalies(grouped, detectors, window_size) -> pd.DataFrame:
    index_cols = [
        "subject_id",
        "study_day",
        "window_size",
        "detector",
    ]
    _, subject_data = grouped
    ads = []
    for detector in detectors:
        sd_copy: pd.DataFrame = subject_data.copy()
        re = detector.getReconstructionError(subject_data)
        if sd_copy.shape[0] != re.shape[0]:
            raise ValueError("Shape mismatch")
        sd_copy = sd_copy.merge(re, validate="1:1", how="inner")
        sd_copy["anomaly"] = detector.labelAnomaly(
            subject_data, recalc_re=False
        )
        sd_copy["detector"] = detector.name
        sd_copy["window_size"] = window_size
        ads.append(sd_copy)
    return pd.concat(ads).set_index(index_cols)

if __name__ == "__main__":
    start = time.perf_counter()

    folder = Path("cache")
    fname = f"{EXPERIMENT}.csv"
    fpath = Path(folder, fname)

    # Load OPTIMA processed healthkit features
    folder = Path("cache")
    dataset = datasets.OPTIMA(
        data_path="/Users/sakre/Data/OPTIMA/OPTIMA_HealthKit/hk_features_Sep13-2023.csv"
    )
    data = dataset.data
    sleep_features = [
        "sleep_sleepDuration_day",
        "sleep_sleepHR_day",
        "sleep_sleepHRV_day",
        "sleep_bedrestDuration_day",
        "sleep_sleepEfficiency_day",
        "sleep_sleepOnsetLatency_day",
        "sleep_bedrestOnsetHours_day",
        "sleep_bedrestOffsetHours_day",
        "sleep_sleepOnsetHours_day",
        "sleep_sleepOffsetHours_day",
        "sleep_Awake_sum",
    ]
    feat_map = {s: s.split('_')[1] for s in sleep_features}
    features_short = list(feat_map.values())
    data = data.rename(columns=feat_map)
    data.loc[data['sleepEfficiency'] == 0, 'sleepEfficiency'] = np.nan
    data.loc[data['sleepDuration'] == 0, 'sleepDuration'] = np.nan

    # Impute missing values
    print("\nImputing dataset")
    imputer = SimpleImputer(strategy='mean', keep_empty_features=True)
    imputed = impute.rollingImpute(
        data=data,
        features=features_short,
        min_days=3,
        imputer=imputer,
        skip_all_nan=True,
        num_cpus=NUM_CPUS,
        drop_any_nan=True
    )
    sleep_features = features_short

    anomalies_detected_list = []
    print("\nRunning anomaly detection in different conditions...")
    conditions = list(product(WINDOW_SIZES, N_COMPONENTS))
    for i, (window_size, components) in enumerate(conditions):
        # Get anomaly detectors with different parameters
        detectors = []
        baseline = anomaly_detection.BaseRollingAnomalyDetector(
            features=sleep_features,
            window_size=window_size,
            max_missing_days=MAX_MISSING_DAYS,
            re_std_threshold=1.65,
            re_abs_threshold=.4,
        )
        if components == N_COMPONENTS[0]:
            detectors.append(baseline)

        if window_size > components:
            pc_detectors = [
                anomaly_detection.PCARollingAnomalyDetector,
                anomaly_detection.NMFRollingAnomalyDetector,
                #anomaly_detection.PCAGridRollingAnomalyDetector,
            ]
            for pc_detector in pc_detectors:
                detectors.append(
                    pc_detector(
                        features=sleep_features,
                        window_size=window_size,
                        max_missing_days=MAX_MISSING_DAYS,
                        n_components=components,
                        re_std_threshold=1,
                        re_abs_threshold=.4,
                    )
                )
        if len(detectors) == 0:
            continue
        dnames = [d.name for d in detectors]

        # Run anomaly detection
        print(
            f"\t {i+1} of {len(conditions)}: window_size: {window_size}, n_components: {components}"
        )



        ad = []
        # Run AD per subject
        if PARALLEL:
            partialDA = partial(detectAnomalies, detectors=detectors, window_size=window_size)
            ad = p_map(partialDA, imputed.groupby("subject_id"), num_cpus=NUM_CPUS)
            ad = pd.concat(ad)
        else:
            for s in imputed.groupby("subject_id"):
                ad.append(detectAnomalies(s, detectors, window_size))
            ad = pd.concat(ad)

        # Add anomalies to full list
        anomalies_detected_list.append(ad)

    anomalies_detected = pd.concat(anomalies_detected_list).reset_index()
    print("\nSaving results to", fpath)
    anomalies_detected.to_csv(fpath, index=False)
    stop = time.perf_counter()
    print(f"\nCompleted in {(stop - start)/60:0.2f} minutes")
