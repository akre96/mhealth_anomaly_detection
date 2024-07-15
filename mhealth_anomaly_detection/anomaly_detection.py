import numpy as np
from numpy.typing import ArrayLike
from typing import Literal
import pandas as pd
from sklearn.decomposition import PCA, NMF
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

from mhealth_anomaly_detection.PCAgrid import PCAgrid


# Base detector takes rolling mean of window size per feature
class BaseRollingAnomalyDetector:
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        re_std_threshold: float = 1.65,
        re_abs_threshold: float = .4,
        remove_past_anomalies: bool = False,
        n_components: int = 0,
    ):
        self.window_size = window_size
        self.max_missing_days = max_missing_days

        # Days of enough reconstruction error to calculate anomaly
        # decision. If window_size = 7, max_missing_days = 2, then
        # min_periods = 5, requires 7 days of data, but can have 2
        # missing days
        self.min_periods = self.window_size - self.max_missing_days
        if self.min_periods < 1:
            self.min_periods = 1

        self.features = features
        self.re_abs_threshold = re_abs_threshold
        self.re_std_threshold = re_std_threshold
        self.reconstruction_error: pd.DataFrame = pd.DataFrame()
        self.name = "RollingMean"
        self.remove_past_anomalies = remove_past_anomalies
        self.scaler = MinMaxScaler()

    @staticmethod
    def validateInputData(subject_data: pd.DataFrame) -> None:
        # data validation
        if "subject_id" not in subject_data:
            raise ValueError("Subject data must have column subject_id")
        if subject_data.subject_id.nunique() != 1:
            raise ValueError(
                "Subject data must have only one unique subject_id"
            )

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
        subject_data = subject_data.sort_values(by="study_day").copy()

        # initialize columns names with features + _re
        df_cols = ["{}_re".format(feature) for feature in self.features]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index,
        )

        # Calculate reconstruction error for each day
        subject_data["anomaly_label"] = np.zeros(subject_data.shape[0]).astype(
            bool
        )
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i >= (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size : i].dropna(
                    subset=self.features
                )

                if train.empty:
                    continue

                if self.remove_past_anomalies:
                    train = train[~train["anomaly_label"]]

                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.scaler.fit(train[self.features])
                train[self.features] = train[self.features]
                # Training set + next day
                X = pd.DataFrame(
                    self.scaler.transform(
                        subject_data.iloc[i - self.window_size : i + 1][
                            self.features
                        ]
                    )
                )

                reconstruction = (
                    pd.DataFrame(self.scaler.transform(train[self.features]))
                    .rolling(
                        window=self.window_size,
                    )
                    .mean()
                )
                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = np.abs(reconstruction.iloc[-1] - X.iloc[-1])

                # Clip reconstruction error to 10
                re_df[re_df > 1] = 1
                total_re = re_df[df_cols].copy().sum(axis=1, min_count=1)
                use_re = total_re[
                    (total_re.index >= (i - self.window_size))
                    & (total_re.index <= i)
                ]
                if self.remove_past_anomalies:
                    use_re = total_re[
                        (total_re.index >= (i - self.window_size))
                        & (total_re.index <= i)
                        & (~subject_data["anomaly_label"])
                    ]
                    subject_data.loc[
                        i, "anomaly_label"
                    ] = self.anomalyDecision(use_re)

        # Normalize to # of features
        re_df["total_re"] = re_df[df_cols].sum(axis=1, min_count=1) / float(len(
            self.features
        ))

        re_df["study_day"] = subject_data["study_day"]
        self.reconstruction_error = re_df
        return re_df

    def getContinuous(self, subject_data, recalc_re: bool = True) -> pd.Series:
        re_df = self.reconstruction_error
        if self.reconstruction_error.empty or recalc_re:
            re_df = self.getReconstructionError(subject_data)

        return re_df["total_re"]

    def anomalyDecision(self, continuous_output: pd.Series) -> bool:
        anomaly_threshold = (
            continuous_output.rolling(
                window=self.window_size,
                min_periods=self.min_periods,
                closed="left",
            ).mean()
            + self.re_std_threshold
            * continuous_output.rolling(
                window=self.window_size,
                min_periods=self.min_periods,
                closed="left",
            ).std()
        )
        return (continuous_output.iloc[-1] > anomaly_threshold.iloc[-1]) | (continuous_output.iloc[-1] > self.re_abs_threshold)

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
                min_periods=self.min_periods,
                closed="left",
            )
            .mean()
            + self.re_std_threshold
            * re_df["total_re"]
            .rolling(
                window=self.window_size,
                min_periods=self.min_periods,
                closed="left",
            )
            .std()
        )
        re_df["threshold"] = anomaly_threshold
        return (re_df["total_re"] > re_df["threshold"]) | (re_df["total_re"] > self.re_abs_threshold)


class PCARollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        re_std_threshold: float = 1.65,
        re_abs_threshold: float = .4,
        remove_past_anomalies: bool = False,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
            re_std_threshold,
            re_abs_threshold,
            remove_past_anomalies,
            n_components
        )
        self.min_features_changing = 1
        self.n_components = n_components
        self.model = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("pca", PCA(n_components=n_components, whiten=True)),
            ]
        )
        self.named_step = "pca"
        str_nc = str(n_components)
        if n_components < 10:
            str_nc = f"00{str_nc}"
        elif n_components < 100:
            str_nc = f"0{str_nc}"

        self.name = "PCA" + "_" + str_nc

    @ignore_warnings(category=ConvergenceWarning)
    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # Make sure input data is valid
        self.validateInputData(subject_data)

        # Sort values
        subject_data = subject_data.sort_values(by="study_day").copy()

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
        subject_data["anomaly_label"] = np.zeros(subject_data.shape[0]).astype(
            bool
        )
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i >= (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size : i].dropna(
                    subset=self.features
                )

                if self.remove_past_anomalies:
                    train = train[train["anomaly_label"] == False]

                if (
                    train.shape[0] < (self.window_size - self.max_missing_days)
                ) or train.empty:
                    continue

                # If 0 variation across any variables skip
                n_params_changing = np.sum(
                    (train[self.features].dropna().diff().abs().sum() > 0)
                )
                if n_params_changing < self.min_features_changing:
                    continue

                self.model.fit(train[self.features])
                pca_components[i, :, :] = self.model.named_steps[
                    self.named_step
                ].components_

                # Training set + next day
                X = subject_data.iloc[i - self.window_size : i + 1][
                    self.features
                ]

                # If out of training day has null values skip
                if np.any(X.iloc[-1].isnull()):
                    continue

                reconstruction = pd.DataFrame(
                    self.model.inverse_transform(
                        self.model.transform(X.dropna())
                    ),
                    columns=self.features,
                )
                # Reconstruction error for out-of-training day kept
                X_scaled = self.model.named_steps["scaler"].transform(
                    X.dropna()
                )
                recon_scaled = self.model.named_steps["scaler"].transform(
                    reconstruction
                )
                re_df.iloc[i] = (np.abs(X_scaled - recon_scaled))[-1]

                # Clip reconstruction error between 10e-10 to 10
                re_df[re_df > 1] = 1
                re_df[re_df.abs() < 1e-14] = 0
                if self.remove_past_anomalies:
                    total_re = re_df[df_cols].sum(axis=1, min_count=1)
                    use_re = total_re[
                        (total_re.index >= (i - self.window_size))
                        & (total_re.index <= i)
                        & (~subject_data["anomaly_label"])
                    ]
                    subject_data.loc[
                        i, "anomaly_label"
                    ] = self.anomalyDecision(use_re)

        re_df["total_re"] = re_df.sum(axis=1, min_count=1) / float(len(
            self.features
        ))
        re_df["study_day"] = subject_data["study_day"]

        # Store pca components in detector class
        self.components: ArrayLike = pca_components
        self.reconstruction_error = re_df

        return re_df


class NMFRollingAnomalyDetector(PCARollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        re_std_threshold: float = 1.65,
        re_abs_threshold: float = .4,
        remove_past_anomalies: bool = False,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
            n_components,
            re_std_threshold,
            re_abs_threshold,
            remove_past_anomalies,
        )
        self.model = Pipeline(
            [
                ("scaler", NMFScaler()),
                ("nmf", NMF(n_components=n_components, max_iter=1000)),
            ]
        )
        self.min_features_changing = 3
        str_nc = str(n_components)
        self.named_step = "nmf"

        if n_components < 10:
            str_nc = f"00{str_nc}"
        elif n_components < 100:
            str_nc = f"0{str_nc}"

        self.name = "NMF" + "_" + str_nc


class PCAGridRollingAnomalyDetector(PCARollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        re_std_threshold: float = 1.65,
        re_abs_threshold: float = .4,
        remove_past_anomalies: bool = False,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
            n_components,
            re_std_threshold,
            re_abs_threshold,
            remove_past_anomalies,
        )
        self.model = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("PCAgrid", PCAgrid(n_components=n_components)),
            ]
        )
        str_nc = str(n_components)
        self.named_step = "PCAgrid"

        if n_components < 10:
            str_nc = f"00{str_nc}"
        elif n_components < 100:
            str_nc = f"0{str_nc}"
        else:
            str_nc = str(n_components)

        self.name = "PCAgrid" + "_" + str_nc


class SVMRollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
        kernel: Literal[
            "linear", "poly", "rbf", "sigmoid", "precomputed"
        ] = "rbf",
        remove_past_anomalies: bool = False,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
            remove_past_anomalies=remove_past_anomalies,
        )
        self.n_components = n_components
        self.model = Pipeline(
            [
                ("scaler", MinMaxScaler()),
                ("svm", OneClassSVM(degree=n_components, kernel=kernel)),
            ]
        )
        self.name = "SVM" + "_" + str(kernel)

    def getReconstructionError(
        self, subject_data: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError(
            f"{self.name} does not have reconstruction error"
        )

    def labelAnomaly(
        self, subject_data: pd.DataFrame, continuous: bool = False, **kwargs
    ) -> pd.Series:
        anomaly_continuous = np.full(subject_data.shape[0], 0)
        anomaly_labels = np.full(subject_data.shape[0], 0)
        # Predict if last day of window anomalous
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                if self.remove_past_anomalies:
                    train = subject_data.iloc[i - self.window_size : i].dropna(
                        subset=self.features
                    )
                    train = train[
                        ~anomaly_labels[i - self.window_size : i].astype(bool)
                    ]
                else:
                    train = subject_data.iloc[i - self.window_size : i].dropna(
                        subset=self.features
                    )
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.model.fit(train[self.features])

                # Training set + next day
                X = subject_data.iloc[i - self.window_size : i + 1][
                    self.features
                ]

                # If out of training day has null values skip
                if np.any(X.iloc[-1].isnull()):
                    continue

                anomaly_continuous[i] = self.model.score_samples(X.dropna())[
                    -1
                ]
                anomaly_labels[i] = self.model.predict(X.dropna())[-1] == 1
        if not continuous:
            anomaly_labels[anomaly_labels == -1.0] = 0
            return pd.Series(anomaly_labels == 1, index=subject_data.index)
        return pd.Series(anomaly_continuous, index=subject_data.index)

    def getContinuous(self, subject_data, **kwargs) -> pd.Series:
        return self.labelAnomaly(subject_data, continuous=True)


class IFRollingAnomalyDetector(SVMRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        remove_past_anomalies: bool = False,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days,
            remove_past_anomalies=remove_past_anomalies,
        )
        self.model = IsolationForest()
        self.name = "IsolationForest"
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features


# MinMax scaler but transform returns 0 for values < 0
class NMFScaler(MinMaxScaler):
    def transform(self, X):
        X = super().transform(X)
        X[X < 0] = 0
        return X


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
