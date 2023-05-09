import pandas as pd
import numpy as np
from typing import Dict, Any
from numpy.typing import ArrayLike
from pathlib import Path
import json


class BaseDailyDataSimulator:
    """Base class for daily data simulation"""

    def __init__(
        self,
        feature_params: Dict[str, Dict],
        n_days: int = 60,
        n_subjects: int = 2,
        sim_type: str = "base",
        cache_simulation: bool = True,
    ):
        self.feature_params = feature_params
        self.n_days = n_days
        self.n_subjects = n_subjects
        self.sim_type = sim_type
        self.cache_simulation = cache_simulation
        self.added_features = []

    # TODO: Test that this works
    def addCorrelatedFeatures(
        self,
        data: pd.DataFrame,
        n_feats: int,
        noise_scale: float = 0.1,
    ) -> pd.DataFrame:
        """Create features correlated to existing features

        Args:
            data (pd.DataFrame): Generated dataset
            n_feats (int): # of additional features per existing one
            noise_scale (float, optional): scale of noise relative to feature mean. Defaults to 0.1.

        Returns:
            pd.DataFrame: data with n_feats * len(feature_params) additional features
        """
        data_added = data.copy()
        for feature, params in self.feature_params.items():
            for i in range(n_feats):
                f_name = f"{feature}_corr_{i}"
                f_noise = np.random.normal(
                    0, params["mean"] * noise_scale, data.shape[0]
                )
                data_added[f_name] = data[feature] + f_noise
                self.added_features.append(f_name)
        return data_added

    def addReLuFeatures(
        self,
        data: pd.DataFrame,
        n_feats: int,
    ) -> pd.DataFrame:
        data_added = data.copy()
        for feature, params in self.feature_params.items():
            for i in range(n_feats):
                f_name = f"{feature}_relu_{i}"
                data_added[f_name] = data[feature] * (
                    data[feature] > params["mean"]
                )
                self.added_features.append(f_name)
        return data_added

    @staticmethod
    def genDailyFeature(
        history: ArrayLike,
        std: float,
        max: float,
        min: float = 0,
        init_value: float = None,
    ) -> float:
        """Generate daily feature from historical data.

        Args:
            history (np.array): historical values, NaNs ignored
            std (float): Feature standard deviation
            max (float): Max feature value
            min (float, optional): Minimum feature value. Defaults to 0.
            init_value (float, optional): Initialization value if no history.

        Raises:
            ValueError: Requires either history or init_value be set

        Returns:
            float: next days feature value
        """
        # Center next value on historical mean, ignore NaNs
        if np.any(~np.isnan(history)):
            loc = np.nanmean(history)
        elif init_value is not None:
            loc = init_value
        else:
            raise ValueError("Initial value or historical values must be set")

        # Generate next days feature value from random normal distribution
        next_val = np.random.normal(
            loc=loc,
            scale=std,
        )

        # Limit boundary of feature
        if next_val > max:
            return max
        if next_val < min:
            return min

        return next_val

    def calcSadEMA(self, subject_data: pd.DataFrame):
        """Calculates daily sad mood from daily features

        Args:
            subject_data (pd.DataFrame): daily data

        Returns:
            pd.Series: simulated EMA Sad choices response
        """
        feature_df = subject_data[list(self.feature_params.keys())]

        # Set features all between 0 and 1
        norm_features = (feature_df - feature_df.min()) / (
            feature_df.max() - feature_df.min()
        )

        # Each feature weighted equally for depressed mood
        sad_ema = norm_features[list(self.feature_params.keys())].sum(axis=1)

        # daily depressed mood is a simple summation scaled between 0 and 3
        sad_ema = 3 * (
            (sad_ema - sad_ema.min()) / (sad_ema.max() - sad_ema.min())
        )
        sad_ema[sad_ema > 3] = 3
        sad_ema = np.round(sad_ema)
        return sad_ema

    def generateSubjectData(
        self,
        subject_id: str,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        """Generate daily features for 1 subject

        Args:
            subject_id (str): name for subject_id column
            random_seed (int, optional): Seed for simulation. Defaults to 0.

        Returns:
            pd.DataFrame: _description_
        """
        np.random.seed(random_seed)
        data_dict = {
            "subject_id": [subject_id] * self.n_days,
            "study_day": range(self.n_days),
        }
        for feature, params in self.feature_params.items():
            f_data = np.full((self.n_days), np.nan)
            for i in range(self.n_days):
                if i == 0:
                    history = [params["mean"]]
                else:
                    history = f_data[i - 1 : (i + params["history_len"] - 1)]
                f_data[i] = self.genDailyFeature(
                    history=history,
                    std=params["std"],
                    max=params["max"],
                    min=params["min"],
                )
            data_dict[feature] = f_data
        return pd.DataFrame(data_dict)

    def genDatasetFilename(self) -> str:
        return (
            "_".join(
                [
                    self.sim_type,
                    "nSubject-" + str(self.n_subjects),
                    "nDay-" + str(self.n_days),
                ]
            )
            + ".csv"
        )

    def simulateData(self, use_cache: bool = True) -> pd.DataFrame:
        """Wrapper function to simulate dataset

        Args:
            use_cache (bool, optional): Use cached file if it exists. Defaults to True.

        Returns:
            pd.DataFrame: simulated dataset
        """
        sim_data = []
        out_path = Path("cache", self.genDatasetFilename())

        # Use cached data if available
        if out_path.is_file() and use_cache:
            print(out_path, "exists. Using cached dataset")
            return pd.read_csv(out_path)

        else:
            # Simulate data
            for i in range(self.n_subjects):
                subject_data = self.generateSubjectData(
                    subject_id="SID_" + str(i), random_seed=i
                )
                subject_data["ema_sad_choices"] = self.calcSadEMA(subject_data)
                sim_data.append(subject_data)

            sim_data = pd.concat(sim_data)

            # Save data
            if self.cache_simulation:
                sim_data.to_csv(out_path, index=False)
                print("\tSaved data to:", out_path)
            return sim_data

    def addPointAnomalies(
        self,
        data: pd.DataFrame,
        anomaly_period: int = 7,
        anomaly_std_scale: float = 2.0,
    ) -> pd.DataFrame:
        """Add point anomalies to simulated data

        Args:
            data (pd.DataFrame): Simulated data
            anomaly_period (int, optional): Number of days between anomalies. Defaults to 7.

        Returns:
            pd.DataFrame: Simulated data with anomalies
        """
        data = data.copy()
        for feature, params in self.feature_params.items():
            for i in range(0, self.n_days, anomaly_period):
                if i == 0:
                    continue
                data.loc[
                    (data["study_day"] == i),
                    feature,
                ] += np.random.normal(
                    loc=0, scale=params["std"] * anomaly_std_scale
                )
        out_path = Path("cache", self.genDatasetFilename())
        if self.cache_simulation:
            data.to_csv(out_path, index=False)
            print("\tSaved data to:", out_path)
        return data


# TODO: Create tests showing that this works
class RandomAnomalySimulator(BaseDailyDataSimulator):
    """Daily data simulator which adds anomalies at a set frequency
    Anomalies are defined as days where the feature value is not tied to historical
    data and has a higher standard deviation
    """

    def __init__(
        self,
        feature_params: Dict[str, Dict],
        n_days: int = 60,
        n_subjects: int = 2,
        sim_type: str = "weeklyAnomaly",
        cache_simulation: bool = True,
    ):
        BaseDailyDataSimulator.__init__(
            self,
            feature_params,
            n_days,
            n_subjects,
            sim_type,
            cache_simulation,
        )
        for feature, params in self.feature_params.items():
            if ("anomaly_frequency" not in params.keys()) or (
                "anomaly_std_scale" not in params.keys()
            ):
                raise ValueError(
                    feature + " Anomaly frequency and scale not specified"
                )

    def generateSubjectData(
        self,
        subject_id: str,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        """Generate daily features for 1 subject with anomalies

        Args:
            subject_id (str): name for subject_id column
            random_seed (int, optional): Seed for simulation. Defaults to 0.

        Returns:
            pd.DataFrame: _description_
        """
        np.random.seed(random_seed)
        data_dict = {
            "subject_id": [subject_id] * self.n_days,
            "study_day": range(self.n_days),
        }
        for feature, params in self.feature_params.items():
            f_data = np.full((self.n_days), np.nan)
            for i in range(self.n_days):
                if i == 0:
                    # Random starting point
                    f_data[i] = (
                        params["max"] - params["min"]
                    ) * np.random.random() + params["min"]
                    continue
                elif params["history_len"] == 0:
                    history = [params["mean"]]
                else:
                    history = f_data[i - 1 : (i + params["history_len"] - 1)]

                # If anomaly day -> increase std and set around mean
                is_anomaly_day = (params["anomaly_frequency"] > 0) and (
                    not (i % params["anomaly_frequency"])
                )
                if is_anomaly_day:
                    f_data[i] = (
                        params["max"] - params["min"]
                    ) * np.random.random() + params["min"]
                else:
                    f_data[i] = self.genDailyFeature(
                        history=history,
                        std=params["std"],
                        max=params["max"],
                        min=params["min"],
                    )
            data_dict[feature] = f_data
        return pd.DataFrame(data_dict)


if __name__ == "__main__":
    n_subjects = 2
    n_days = 60

    with open("lib/feature_parameters.json", "r") as fp:
        all_feature_params = json.load(fp)

    for sim_type, feature_params in all_feature_params.items():
        if sim_type == "base":
            print("Simulating base with no anomaly")
            simulator = BaseDailyDataSimulator(
                feature_params=all_feature_params[sim_type],
                n_days=n_days,
                n_subjects=n_subjects,
                sim_type=sim_type,
            )
            print("Simulating base with point anomaly")
            simulator = BaseDailyDataSimulator(
                feature_params=all_feature_params[sim_type],
                n_days=n_days,
                n_subjects=n_subjects,
                sim_type=sim_type+'PointAnomaly',
            )
            data = simulator.addPointAnomalies(simulator.simulateData())
            print("\nPreview of data: ")
            print(data.head(n=10))

        else:
            print("Simulating", sim_type)
            simulator = RandomAnomalySimulator(
                feature_params=feature_params,
                n_days=n_days,
                n_subjects=n_subjects,
                sim_type=sim_type,
            )
            data = simulator.simulateData(use_cache=False)
            print("\nPreview of data: ")
            print(data.head(n=10))
