import pandas as pd
import numpy as np
from typing import Dict, Any
from pathlib import Path
import json


class BaseDailyDataSimulator:
    """ Base class for daily data simulation
    """
    def __init__(
        self,
        feature_params: Dict[str, Dict],
        n_days: int = 60,
        n_subjects: int = 2,
        sim_type: str = 'base'
    ):
        self.feature_params = feature_params
        self.n_days = n_days
        self.n_subjects = n_subjects
        self.sim_type = sim_type

    @staticmethod
    def genDailyFeature(
        history: np.array,
        std: float,
        max: float,
        min: float = 0,
        init_value: float = None,
    ) -> float:
        """ Generate daily feature from historical data.

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
        if np.any(history):
            loc = np.nanmean(history)
        elif init_value is not None:
            loc = init_value
        else:
            raise ValueError('Initial value or historical values must be set')

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
        """ Calculates daily sad mood from daily features

        Args:
            subject_data (pd.DataFrame): daily data

        Returns:
            pd.Series: simulated EMA Sad choices response
        """
        feature_df = subject_data[self.feature_params.keys()]

        # Set features all between 0 and 1
        norm_features = (feature_df - feature_df.min()) /\
            (feature_df.max() - feature_df.min())

        # Each feature weighted equally for depressed mood
        sad_ema = norm_features['hr_resting'] \
            - norm_features['active_energy_burned'] \
            - norm_features['hrv_mean'] \
            - norm_features['total_sleep_time']

        # daily depressed mood is a simple summation scaled between 0 and 3
        sad_ema = 3 * (
            (sad_ema - sad_ema.min()) /
            (sad_ema.max() - sad_ema.min())
        )
        sad_ema[sad_ema > 3] = 3
        sad_ema = np.round(sad_ema)
        return sad_ema

    def generateSubjectData(
        self,
        subject_id: str,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        """ Generate daily features for 1 subject

        Args:
            subject_id (str): name for subject_id column
            random_seed (int, optional): Seed for simulation. Defaults to 0.

        Returns:
            pd.DataFrame: _description_
        """
        np.random.seed(random_seed)
        data_dict = {
            'subject_id': [subject_id] * self.n_days,
            'study_day': range(self.n_days),
        }
        for feature, params in self.feature_params.items():
            f_data = np.full((self.n_days), np.nan)
            for i in range(self.n_days):
                if i == 0:
                    history = [params['mean']]
                else:
                    history = f_data[i-1:(i+params['history_len']-1)]
                f_data[i] = self.genDailyFeature(
                    history=history,
                    std=params['std'],
                    max=params['max'],
                    min=params['min'],
                )
            data_dict[feature] = f_data
        return pd.DataFrame(data_dict)

    def genDatasetFilename(self) -> str:
        return '_'.join([
                self.sim_type,
                'nSubject-'+str(self.n_subjects),
                'nDay-'+str(self.n_days),
            ]) + '.csv'

    def simulateData(self, use_cache: bool = True) -> pd.DataFrame:
        """ Wrapper function to simulate dataset

        Args:
            use_cache (bool, optional): Use cached file if it exists. Defaults to True.

        Returns:
            pd.DataFrame: simulated dataset
        """
        sim_data = []
        out_path = Path(
            'cache',
            self.genDatasetFilename()
        )

        # Use cached data if available
        if out_path.is_file() and use_cache:
            print(out_path, 'exists. Using cached dataset')
            return pd.read_csv(out_path)

        else:
            print('Simulating data...')
            # Simulate data
            for i in range(self.n_subjects):
                subject_data = self.generateSubjectData(
                        subject_id='SID_'+str(i),
                        random_seed=i
                    )
                subject_data['ema_sad_choices'] = self.calcSadEMA(subject_data)
                sim_data.append(subject_data)

            # Save data
            sim_data = pd.concat(sim_data)
            sim_data.to_csv(out_path, index=False)
            print('\tSaved data to:', out_path)
            return sim_data


# TODO: Create tests showing that this works
class RandomAnomalySimulator(BaseDailyDataSimulator):
    """ Daily data simulator which adds anomalies at a set frequency
    """
    def __init__(
        self,
        feature_params: Dict[str, Dict],
        n_days: int = 60,
        n_subjects: int = 2,
        sim_type: str = 'base',
        anomaly_frequency: int = 7,
        anomaly_std_scale: float = 2,
    ):
        BaseDailyDataSimulator.__init__(
            self,
            feature_params,
            n_days,
            n_subjects,
            sim_type,
        )
        self.anomaly_frequency = anomaly_frequency
        self.anomaly_std_scale = anomaly_std_scale

    def generateSubjectData(
        self,
        subject_id: str,
        random_seed: int = 0,
    ) -> pd.DataFrame:
        """ Generate daily features for 1 subject with anomalies

        Args:
            subject_id (str): name for subject_id column
            random_seed (int, optional): Seed for simulation. Defaults to 0.

        Returns:
            pd.DataFrame: _description_
        """
        np.random.seed(random_seed)
        data_dict = {
            'subject_id': [subject_id] * self.n_days,
            'study_day': range(self.n_days),
        }
        for feature, params in self.feature_params.items():
            f_data = np.full((self.n_days), np.nan)
            for i in range(self.n_days):
                if i == 0:
                    history = [params['mean']]
                else:
                    history = f_data[i-1:(i+params['history_len']-1)]
                
                # If anomaly day -> increase std and set around mean
                is_anomaly_day = (
                    (not (i % self.anomaly_frequency)) and
                    (i >= self.anomaly_frequency)
                )
                if is_anomaly_day:
                    f_data[i] = self.genDailyFeature(
                        history=None,
                        std=params['std'] * self.anomaly_std_scale,
                        max=params['max'],
                        min=params['min'],
                        init_value=params['mean'],
                    )
                else:
                    f_data[i] = self.genDailyFeature(
                        history=history,
                        std=params['std'],
                        max=params['max'],
                        min=params['min'],
                    )
            data_dict[feature] = f_data
        return pd.DataFrame(data_dict)

    def genDatasetFilename(self) -> str:
        return '_'.join([
                self.sim_type,
                'nSubject-'+str(self.n_subjects),
                'nDay-'+str(self.n_days),
                'anomalyFreq-'+str(self.anomaly_frequency),
                'anomalyStdScale-'+str(self.anomaly_std_scale),
            ]) + '.csv'


if __name__ == '__main__':
    n_subjects = 2
    n_days = 60
    sim_type = 'base'

    with open('lib/feature_parameters.json', 'r') as fp:
        feature_params = json.load(fp)[sim_type]

    print('Simulating base with no anomaly')
    simulator = BaseDailyDataSimulator(
        feature_params=feature_params,
        n_days=n_days,
        n_subjects=n_subjects,
        sim_type=sim_type
    )
    data = simulator.simulateData()
    print('\nPreview of data: ')
    print(data.head(n=10))

    print('Simulating base with weekly anomalies')
    simulator = RandomAnomalySimulator(
        feature_params=feature_params,
        n_days=n_days,
        n_subjects=n_subjects,
        sim_type=sim_type,
        anomaly_frequency=7,
        anomaly_std_scale=2,
    )
    data = simulator.simulateData(use_cache=False)
    print('\nPreview of data: ')
    print(data.head(n=10))