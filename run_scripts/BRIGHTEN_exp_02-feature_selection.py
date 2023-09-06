import pandas as pd
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm

# from pandarallel import pandarallel
import argparse

from mhealth_anomaly_detection import (
    datasets,
    anomaly_detection as ad,
    impute,
)
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer

STUDY = "BRIGHTEN_v2"
EXP = "exp02"
out_path = Path("cache", f"{STUDY}_{EXP}.csv")

AD_PARAMS = [
    {
        "window_size": 7,
        "max_missing_days": 2,
    },
    {
        "window_size": 14,
        "max_missing_days": 2,
    },
    {
        "window_size": 28,
        "max_missing_days": 4,
    },
]


def run_anomaly_detection(input) -> pd.DataFrame:
    data, detector = input

    def ad_apply(group):
        re = detector.getReconstructionError(group)
        group["anomaly"] = detector.labelAnomaly(group)
        group[re.columns] = re
        return group

    """
    try:
        labeled = data.groupby("subject_id", group_keys=False).parallel_apply(
            ad_apply
        )
    except TypeError:
        print(
            "Error in parallel apply, retrying with single-threaded apply",
            detector.name,
        )
        labeled = data.groupby("subject_id", group_keys=False).apply(ad_apply)
    """
    labeled = data.groupby("subject_id", group_keys=False).apply(ad_apply)
    labeled["detector"] = detector.name
    labeled["window_size"] = detector.window_size
    return labeled


if __name__ == "__main__":
    # Get debug flag
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (only 5 subjects, 1 detector)",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=1,
        help="Number of CPUs to use for parallel processing",
    )
    parser.add_argument(
        "--rerun-cache",
        action="store_true",
        help="Don't use cached data",
    )

    args = parser.parse_args()
    num_cpus = args.num_cpus
    rerun_cache = args.rerun_cache

    # pandarallel.initialize(nb_workers=num_cpus, progress_bar=False)
    # Ignore divide by 0 error -> expected and happens in PCA
    np.seterr(divide="ignore", invalid="ignore")

    detectors = [
        ad.PCAGridRollingAnomalyDetector,
        ad.BaseRollingAnomalyDetector,
        ad.PCARollingAnomalyDetector,
        ad.NMFRollingAnomalyDetector,
    ]

    data_path = Path("~/Data/mHealth_external_datasets/BRIGHTEN/").expanduser()
    passive_features = [
        "mobility",
        # 'phone_communication', # Phone comms only really available for Android
        # 'weather',
    ]

    dataset = datasets.BRIGHTEN_v2(
        data_path=data_path, feature_types=passive_features
    )
    data = dataset.data
    features = dataset.features

    if args.debug:
        num_cpus = 1
        out_path = Path("cache", f"{STUDY}_{EXP}_DEBUG.csv")
        rerun_cache = True
        detectors = [detectors[0]]
        data = data[data.subject_id.isin(data.subject_id.unique()[:5])]
        print("Running in debug mode")

    # Initialize anomaly detectors with parameters
    initialized_detectors = []
    for params in AD_PARAMS:
        for d in detectors:
            initialized_detectors.append(
                d(
                    features=features,
                    remove_past_anomalies=False,
                    re_std_threshold=1.65,
                    n_components=3,
                    **params,
                )
            )

    cache_path = Path("cache", f"{STUDY}_{EXP}_anomaly_labeled.csv")
    if not rerun_cache and cache_path.exists():
        print("Loading from cache...")
        anomaly_labeled = pd.read_csv(cache_path, parse_dates=["date"])
    else:
        # Get training data
        train_test_split = dataset.getTrainTestSplit()
        train_sids = train_test_split[
            train_test_split.batch == 1
        ].subject_id.unique()

        train = data[data.subject_id.isin(train_sids)]

        imputer = IterativeImputer(
            initial_strategy="median",
            keep_empty_features=True,
            skip_complete=True,
        )
        print("Imputing...")
        imputed = impute.rollingImpute(
            train,
            imputer=imputer,
            features=features,
            min_days=3,
            num_cpus=num_cpus,
            skip_all_nan=True,
        )
        print("\tdone")

        print("Running anomaly detection...")
        anomaly_labeled = pd.concat(
            [
                run_anomaly_detection((train, d))
                for d in tqdm(initialized_detectors)
            ]
        )
        # Save to cache
        anomaly_labeled.to_csv(cache_path, index=False)
        print("\tdone")

    print("Comparing anomalies to self_report scores...")
    # Gather Self-Report data
    phq9 = dataset.getSelfReport("phq9")
    phq9["lookback"] = "7d"

    sds = dataset.getSelfReport("SDS")
    sds["lookback"] = "7d"

    """ Not enough data for GAD7 or sleep quality
    gad7 = dataset.getSelfReport("GAD7")
    gad7["lookback"] = "7d"

    sleep_quality = dataset.getSelfReport("sleep_quality")
    sleep_quality["lookback"] = "7d"
    """

    self_reports = phq9.merge(
        sds,
        on=["subject_id", "date", "self_report", "lookback"],
        validate="1:1",
        how="outer",
    )
    self_reports["date"] = pd.to_datetime(self_reports["date"])
    self_report_anomalies = self_reports.copy()

    # Compare to anomalies
    anomaly_cols = []
    for d in initialized_detectors:
        re_cols = [f"{c}_re" for c in dataset.features] + ["total_re"]
        re_d_cols = [f"{d.name}-{d.window_size}_{c}" for c in re_cols]
        self_report_anomalies[f"{d.name}-{d.window_size}_anomaly_count"] = 0
        anomaly_cols.append(f"{d.name}-{d.window_size}_anomaly_count")

    for i, row in self_reports.iterrows():
        response_anomalies = anomaly_labeled.loc[
            (anomaly_labeled.subject_id == row.subject_id)
            & (anomaly_labeled.date < row.date)
            & (anomaly_labeled.date >= (row.date - pd.Timedelta("7d")))
        ]
        for (ws, d), d_df in response_anomalies.groupby(
            ["window_size", "detector"]
        ):
            set_d = 0
            re_d_cols = [f"{d}-{ws}_{c}" for c in re_cols]
            self_report_anomalies.loc[i, f"{d}-{ws}_anomaly_count"] = d_df[
                "anomaly"
            ].sum()
            self_report_anomalies.loc[i, re_d_cols] = (
                d_df[re_cols].mean().to_numpy()
            )

            if not set_d:
                self_report_anomalies.loc[i, "days_with_data"] = d_df.shape[0]
                self_report_anomalies.loc[i, "missing_days"] = (
                    d_df[features].isnull().any(axis=1).sum()
                )
                set_d = 1
    print("\tdone")
    print("Saving to:", out_path)
    self_report_anomalies.to_csv(out_path, index=False)
    print(
        self_report_anomalies[
            ["days_with_data", "missing_days", *anomaly_cols]
        ].describe()
    )
