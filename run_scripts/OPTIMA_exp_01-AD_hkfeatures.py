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

# explicitly require this experimental feature
from sklearn.experimental import enable_iterative_imputer  # noqa

# now you can import normally from sklearn.impute
from sklearn.impute import IterativeImputer

# Meta params
NUM_CPUS = 10
MAX_MISSING_DAYS = 2
EXPERIMENT = "OPTIMA_exp01"

# Anomaly detector Parameters
WINDOW_SIZES = [28]
N_COMPONENTS = [5, 10, 20]
MIN_DAYS = 7
MAX_MISSING_DAYS = 2

# Runtime parameters
USE_CACHE = False
PARALLEL = False

if __name__ == "__main__":
    start = time.perf_counter()

    folder = Path("cache")
    fname = f"{EXPERIMENT}.csv"
    fpath = Path(folder, fname)

    # Load OPTIMA processed healthkit features
    folder = Path("cache")
    dataset = datasets.OPTIMA(
        data_path="~/Data/OPTIMA/hk_features_May15-2023.csv"
    )
    data = dataset.data
    features = dataset.sensor_cols

    # Impute missing values
    print("\nImputing dataset")
    imputer = IterativeImputer(
        initial_strategy="median",
        keep_empty_features=True,
        skip_complete=True,
    )

    imputed = impute.rollingImpute(
        data, features, MIN_DAYS, imputer, num_cpus=NUM_CPUS
    )

    anomalies_detected_list = []
    print("\nRunning anomaly detection in different conditions...")
    conditions = list(product(WINDOW_SIZES, N_COMPONENTS))
    for i, (window_size, components) in enumerate(conditions):
        detectors = []
        baseline = anomaly_detection.BaseRollingAnomalyDetector(
            features=features,
            window_size=window_size,
            max_missing_days=MAX_MISSING_DAYS,
        )
        if components == N_COMPONENTS[0]:
            detectors.append(baseline)

        if window_size > components:
            pc_detectors = [
                anomaly_detection.PCARollingAnomalyDetector,
                anomaly_detection.NMFRollingAnomalyDetector,
            ]
            for pc_detector in pc_detectors:
                detectors.append(
                    pc_detector(
                        features=features,
                        window_size=window_size,
                        max_missing_days=MAX_MISSING_DAYS,
                        n_components=components,
                    )
                )
        if len(detectors) == 0:
            continue
        dnames = [d.name for d in detectors]

        print(
            f"\t {i+1} of {len(conditions)}: window_size: {window_size}, n_components: {components}"
        )

        def detectAnomalies(grouped) -> pd.DataFrame:
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

        if PARALLEL:
            ad = pd.concat(
                p_map(
                    detectAnomalies,
                    imputed.groupby("subject_id"),
                    num_cpus=NUM_CPUS,
                )
            )
        else:
            ad = []
            for s in imputed.groupby("subject_id"):
                ad.append(detectAnomalies(s))
            ad = pd.concat(ad)
        anomalies_detected_list.append(ad)

    anomalies_detected = pd.concat(anomalies_detected_list).reset_index()
    print("\nSaving results to", fpath)
    anomalies_detected.to_csv(fpath, index=False)
    stop = time.perf_counter()
    print(f"\nCompleted in {(stop - start)/60:0.2f} minutes")
