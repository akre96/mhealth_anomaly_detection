import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List
import pandas as pd
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
import scipy.stats as stats
from mhealth_anomaly_detection.onmf import Online_NMF


# Base detector takes rolling mean of window size per feature
class BaseRollingAnomalyDetector:
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        re_std_threshold: float = 2.0,
    ):
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
        self.re_std_threshold = re_std_threshold
        self.reconstruction_error: pd.DataFrame = pd.DataFrame()
        self.name = "RollingMean"

    @staticmethod
    def validateInputData(subject_data: pd.DataFrame) -> None:
        # data validation
        if "subject_id" not in subject_data:
            raise ValueError("Subject data must have column subject_id")
        if subject_data.subject_id.nunique() != 1:
            raise ValueError("Subject data must have only one unique subject_id")

        day_range = subject_data.study_day.max() - subject_data.study_day.min()
        if subject_data.shape[0] != day_range + 1:
            raise ValueError(
                "Subject data must have a row for every day even if missing feature values"
            )

    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # data validation
        self.validateInputData(subject_data)

        # Sort values
        subject_data = subject_data.sort_values(by="study_day")

        # initialize columns names with features + _re
        df_cols = ["{}_re".format(feature) for feature in self.features]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index,
        )

        # reconstruction as rolling window_size mean
        reconstruction = (
            subject_data[self.features]
            .rolling(
                window=self.window_size,
                min_periods=self.window_size - self.max_missing_days,
            )
            .mean()
        )

        re_df[df_cols] = (
            (subject_data[self.features] - reconstruction) / reconstruction
        ).abs()

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10
        re_df["total_re"] = re_df.sum(axis=1, min_count=1)
        re_df["study_day"] = subject_data["study_day"]
        self.reconstruction_error = re_df
        return re_df

    # Return if anomalous day labels
    def labelAnomaly(
        self, subject_data: pd.DataFrame, recalc_re: bool = True
    ) -> pd.Series:
        if self.reconstruction_error.empty or recalc_re:
            re_df = self.getReconstructionError(subject_data)
        else:
            re_df = self.reconstruction_error

        # anomaly as mean + 2*std of reconstruction error
        anomaly_threshold = (
            re_df["total_re"]
            .rolling(
                window=self.window_size,
                min_periods=self.window_size - self.max_missing_days,
            )
            .mean()
            + self.re_std_threshold
            * re_df["total_re"]
            .rolling(
                window=self.window_size,
                min_periods=self.window_size - self.max_missing_days,
            )
            .std()
        )
        return re_df["total_re"] > anomaly_threshold


class PCARollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        re_std_threshold: float = 2,
    ):
        super().__init__(features, window_size, max_missing_days, re_std_threshold)
        self.n_components = n_components
        self.model = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("pca", PCA(n_components=n_components, whiten=True)),
            ]
        )
        str_nc = str(n_components)
        if n_components < 10:
            str_nc = f"00{str_nc}"
        elif n_components < 100:
            str_nc = f"0{str_nc}"

        self.name = "PCA" + "_" + str_nc

    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # Make sure input data is valid
        self.validateInputData(subject_data)

        # Sort values
        subject_data = subject_data.sort_values(by="study_day")

        # initialize columns names with features + _re
        df_cols = ["{}_re".format(feature) for feature in self.features]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index,
        )
        pca_components = np.full(
            (subject_data.shape[0], self.n_components, len(df_cols)), np.nan
        )

        # Calculate reconstruction error for each day
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size : i].dropna(
                    subset=self.features
                )
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.model.fit(train[self.features])
                pca_components[i, :, :] = self.model.named_steps["pca"].components_

                # Training set + next day
                X = subject_data.iloc[i - self.window_size : i + 1][self.features]

                # If out of training day has null values skip
                if np.any(X.iloc[-1].isnull()):
                    continue

                reconstruction = pd.DataFrame(
                    self.model.inverse_transform(self.model.transform(X.dropna())),
                    columns=self.features,
                )
                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = (
                    np.abs(
                        self.model.named_steps["scaler"].transform(X.dropna())
                        - self.model.named_steps["scaler"].transform(reconstruction)
                    )
                )[-1]

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10

        re_df["total_re"] = (re_df).sum(axis=1, min_count=1)
        re_df["study_day"] = subject_data["study_day"]

        # Store pca components in detector class
        self.components: ArrayLike = pca_components
        self.reconstruction_error = re_df

        return re_df


class SVMRollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        kernel: str = "rbf",
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
        )
        self.n_components = n_components
        self.model = Pipeline(
            [
                ("scaler", RobustScaler()),
                ("svm", OneClassSVM(degree=n_components, kernel=kernel)),
            ]
        )
        self.name = "SVM" + "_" + str(kernel)

    def labelAnomaly(self, subject_data: pd.DataFrame) -> NDArray:
        anomaly_labels = np.full(subject_data.shape[0], 0)

        # Predict if last day of window anomalous
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size : i].dropna(
                    subset=self.features
                )
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.model.fit(train[self.features])

                # Training set + next day
                X = subject_data.iloc[i - self.window_size : i + 1][self.features]

                # If out of training day has null values skip
                if np.any(X.iloc[-1].isnull()):
                    continue

                anomaly_labels[i] = self.model.predict(X.dropna())[-1]
        anomaly_labels[anomaly_labels == -1.0] = 0
        anomaly_labels[np.isnan(anomaly_labels)] = 0
        return anomaly_labels == 1


# TODO: Refactor with SK-learn implementation
class NMFRollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        re_std_threshold: float = 2.0,
    ):
        super().__init__(features, window_size, max_missing_days, re_std_threshold)
        self.n_components = n_components
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features

        str_nc = str(n_components)
        if n_components < 10:
            str_nc = f"00{str_nc}"
        elif n_components < 100:
            str_nc = f"0{str_nc}"

        self.name = "NMF" + "_" + str_nc

    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validateInputData(subject_data)
        # Sort values
        subject_data = subject_data.sort_values(by="study_day")

        # initialize columns names with features + _re
        df_cols = ["{}_re".format(feature) for feature in self.features]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index,
        )
        nmf_components = np.full(
            (subject_data.shape[0], len(df_cols), self.n_components), np.nan
        )

        # Calculate reconstruction error for each day
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size : i].dropna(
                    subset=self.features
                )[self.features]
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                scaler = RobustScaler()
                scaler.fit(train)
                model = Online_NMF(
                    scaler.transform(train).T,
                    n_components=self.n_components,
                    iterations=10,
                    batch_size=round(self.window_size / 2),
                )
                W = model.train_dict()
                nmf_components[i, :, :] = W

                # Training set + next day reconstructed
                X = scaler.transform(
                    subject_data.iloc[i - self.window_size : i + 1][
                        self.features
                    ].dropna()
                ).T
                H = model.sparse_code(X, W)
                reconstruction = W @ H

                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = (np.abs(X - reconstruction).T)[-1]

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10

        re_df["total_re"] = (re_df).sum(axis=1, min_count=1)
        re_df["study_day"] = subject_data["study_day"]

        # Store pca components in detector class
        self.components: ArrayLike = nmf_components
        self.reconstruction_error = re_df

        return re_df


class IFRollingAnomalyDetector(SVMRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
    ):
        super().__init__(features, window_size, max_missing_days)
        self.model = IsolationForest()
        self.name = "IsolationForest"
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features


# Calculate accuracy, sensitivity and specificity
def performance_metrics(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str] = [
        "subject_id",
        "history_type",
        "window_size",
        "anomaly_freq",
    ],
) -> pd.DataFrame:
    models = [c.split("_anomaly")[0] for c in anomaly_detector_cols]
    performance_dict = {c: [] for c in groupby_cols}
    performance_dict["model"] = []
    performance_dict["true_positives"] = []
    performance_dict["true_negatives"] = []
    performance_dict["false_positives"] = []
    performance_dict["false_negatives"] = []

    for info, subject_data in tqdm(data.groupby(groupby_cols)):
        subject_data["anomaly"] = subject_data["anomaly"].astype(bool)
        # Fix error if only one groupby item, info is a string, not a tuple[str]
        if type(info) == str:
            info = [info]
        for model in models:
            subject_data[model + "_anomaly"] = subject_data[model + "_anomaly"].astype(
                bool
            )
            for i, val in enumerate(info):
                performance_dict[groupby_cols[i]].append(val)
            performance_dict["model"].append(model)
            performance_dict["true_positives"].append(
                (subject_data["anomaly"] & subject_data[model + "_anomaly"]).sum()
            )
            performance_dict["true_negatives"].append(
                (~subject_data["anomaly"] & ~subject_data[model + "_anomaly"]).sum()
            )
            performance_dict["false_negatives"].append(
                (subject_data["anomaly"] & ~subject_data[model + "_anomaly"]).sum()
            )
            performance_dict["false_positives"].append(
                (~subject_data["anomaly"] & subject_data[model + "_anomaly"]).sum()
            )

    performance_df = pd.DataFrame(performance_dict)
    performance_df["sensitivity"] = performance_df["true_positives"] / (
        performance_df["true_positives"] + performance_df["false_negatives"]
    )
    performance_df["precision"] = performance_df["true_positives"] / (
        performance_df["true_positives"] + performance_df["false_positives"]
    )
    performance_df["specificity"] = performance_df["true_negatives"] / (
        performance_df["true_negatives"] + performance_df["false_positives"]
    )
    performance_df["accuracy"] = performance_df[
        ["true_positives", "true_negatives"]
    ].sum(axis=1) / performance_df[
        ["true_positives", "true_negatives", "false_positives", "false_negatives"]
    ].sum(
        axis=1
    )
    performance_df["F1"] = (
        2
        * performance_df["sensitivity"]
        * performance_df["precision"]
        / performance_df[["sensitivity", "precision"]].sum(axis=1)
    )
    return performance_df


# Find distance of induced anomaly to closest detected anomaly
def distance_real_to_detected_anomaly(
    data: pd.DataFrame, groupby_cols: List[str], anomaly_detector_cols: List[str]
) -> pd.DataFrame:
    anomaly_detector_distances = []

    models = [c.split("_anomaly")[0] for c in anomaly_detector_cols]
    for info, subject_data in tqdm(data.groupby(groupby_cols)):
        for model in models:
            subject_data[model + "_anomaly"] = subject_data[model + "_anomaly"].astype(
                bool
            )
        subject_data["anomaly"] = subject_data["anomaly"].astype(bool)
        # day of detected anomaly
        anomaly_days = {
            c: subject_data.loc[subject_data[c + "_anomaly"], "study_day"].values
            for c in models
        }

        # Actual induced anomalies
        real_anomaly = subject_data.loc[subject_data["anomaly"], "study_day"].values

        # Initialize with NaN values
        anomaly_distance = {c: np.full(real_anomaly.shape, np.nan) for c in models}
        # Calculate distance for each induced anomaly to closest future detected anomaly
        for i in range(real_anomaly.shape[0]):
            for c in models:
                distances = anomaly_days[c] - real_anomaly[i]
                pos_distances = distances[distances >= 0]

                # If no future detected anomaly, fill distance with np.nan
                if np.any(pos_distances >= 0):
                    min_pos = np.min(pos_distances)
                else:
                    min_pos = np.nan
                anomaly_distance[c][i] = min_pos

        anomaly_distance = pd.DataFrame(anomaly_distance)
        anomaly_distance["study_day"] = real_anomaly
        for i, c in enumerate(groupby_cols):
            anomaly_distance[c] = info[i]
        anomaly_detector_distances.append(anomaly_distance)

    # for each detector's anomalies, find closest day of real anomaly on or after detected, and calculate distance
    # If no real anomaly before detected anomaly, distance = np.nan
    anomaly_detector_distance_df = pd.concat(anomaly_detector_distances)
    anomaly_detector_distance_df = anomaly_detector_distance_df.melt(
        id_vars=["study_day", *groupby_cols], var_name="model", value_name="distance"
    )
    return anomaly_detector_distance_df


if __name__ == "__main__":
    import load_refs as lr
    import simulate_daily as sd

    all_feature_params = lr.get_all_feature_params()

    for sim_type, feature_params in all_feature_params.items():
        # Load data
        if "Anomaly" in sim_type:
            simulator = sd.RandomAnomalySimulator(
                feature_params=feature_params,
            )
            dataset = simulator.simulateData()

            for subject, subject_data in dataset.groupby("subject_id"):
                # Detect anomalies
                anomalyDetector = NMFRollingAnomalyDetector(
                    features=feature_params.keys()
                )

                re = anomalyDetector.getReconstructionError(subject_data)
                re["anomaly"] = anomalyDetector.labelAnomaly(re)
                print(re)
                break


def correlateDetectedToOutcome(
    detected_anomalies: pd.DataFrame,
    anomaly_detector_cols: List[str],
    outcome_col: str,
    groupby_cols: List[str],
) -> pd.DataFrame:
    corr_dict = {
        "detector": [],
        "rho": [],
        "p": [],
        "n": [],
        **{inf: [] for inf in groupby_cols},
    }
    for info, i_df in detected_anomalies.groupby(groupby_cols):
        for d in anomaly_detector_cols:
            d_df = i_df[[d, outcome_col]].dropna()
            n = d_df.shape[0]
            if (d_df[d].nunique() == 1) or (d_df[outcome_col].nunique() == 1):
                rho, p = (0, 1)
            else:
                rho, p = stats.spearmanr(d_df[d], d_df[outcome_col])
            corr_dict["detector"].append(d)
            for i in range(len(groupby_cols)):
                corr_dict[groupby_cols[i]].append(info[i])
            corr_dict["n"].append(n)
            corr_dict["rho"].append(rho)
            corr_dict["p"].append(p)
    return pd.DataFrame(corr_dict).rename(
        {c: c.split("_anomaly")[0] for c in anomaly_detector_cols}
    )


# Correlation between # detected and induced anomalies in simulation
def correlateDetectedToInduced(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str],
    corr_across: List[str],
) -> pd.DataFrame:
    for c in corr_across:
        if c not in groupby_cols:
            raise ValueError(f"corr_across value {c} not found in groupby_cols")

    # Ensure bool
    for c in ["anomaly", *anomaly_detector_cols]:
        data[c] = data[c].astype(bool)

    # Anomalies detected per subject/model
    detected_anomalies = (
        data.groupby(groupby_cols)[["anomaly"] + anomaly_detector_cols]
        .sum(numeric_only=False)
        .reset_index()
    )

    return correlateDetectedToOutcome(
        detected_anomalies, anomaly_detector_cols, "anomaly", corr_across
    )
