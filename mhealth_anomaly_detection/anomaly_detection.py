import numpy as np
from numpy.typing import ArrayLike, NDArray
from typing import List
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from mhealth_anomaly_detection.onmf import Online_NMF


# Base detector takes rolling mean of window size per feature
class BaseRollingAnomalyDetector:
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
    ):
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
        self.reconstruction_error: pd.DataFrame = pd.DataFrame()
        self.name = 'RollingMean'
    
    @staticmethod
    def validateInputData(subject_data: pd.DataFrame) -> None:
        # data validation
        if 'subject_id' not in subject_data:
            raise ValueError('Subject data must have column subject_id')
        if subject_data.subject_id.nunique() != 1:
            raise ValueError('Subject data must have only one unique subject_id')

        day_range = subject_data.study_day.max() - subject_data.study_day.min()
        if subject_data.shape[0] != day_range+1:
            raise ValueError('Subject data must have a row for every day even if missing feature values')

    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # data validation
        self.validateInputData(subject_data)
        
        # Sort values
        subject_data = subject_data.sort_values(by='study_day')

        # initialize columns names with features + _re
        df_cols = [
            '{}_re'.format(feature) for feature in self.features
        ]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index
        )

        # reconstruction as rolling window_size mean
        reconstruction = subject_data[self.features].rolling(
            window=self.window_size,
            min_periods=self.window_size - self.max_missing_days,
        ).mean()

        re_df[df_cols] = ((
            subject_data[self.features] - 
            reconstruction
        )/reconstruction).abs()

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10
        re_df['total_re'] = re_df.sum(axis=1, min_count=1)
        re_df['study_day'] = subject_data['study_day']
        self.reconstruction_error = re_df
        return re_df

    # Return if anomalous day labels
    def labelAnomaly(self, subject_data: pd.DataFrame) -> pd.Series:
        if not self.reconstruction_error.empty:
            re_df = self.reconstruction_error
        else:
            re_df = self.getReconstructionError(subject_data)

        # anomaly as mean + 2*std of reconstruction error
        anomaly_threshold = re_df['total_re'].mean() + 2*re_df['total_re'].std()
        return re_df['total_re'] > anomaly_threshold


class PCARollingAnomalyDetector(BaseRollingAnomalyDetector):
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days
        )
        self.n_components = n_components
        self.model = Pipeline([
                    ('scaler', RobustScaler()),
                    ('pca', PCA(n_components=n_components, whiten=True))
                ])
        self.name = 'PCA' + '_' + str(n_components)
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
    
    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        # Make sure input data is valid
        self.validateInputData(subject_data)
        
        # Sort values
        subject_data = subject_data.sort_values(by='study_day')

        # initialize columns names with features + _re
        df_cols = [
            '{}_re'.format(feature) for feature in self.features
        ]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index
        )
        pca_components = np.full(
            (
                subject_data.shape[0], self.n_components, len(df_cols)
            ),
            np.nan
        )

        # Calculate reconstruction error for each day
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size:i].dropna(
                    subset=self.features
                )
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.model.fit(train[self.features])
                pca_components[i, :, :] = self.model.named_steps['pca'].components_

                # Training set + next day
                X = subject_data.iloc[i - self.window_size: i+1][self.features]

                # If out of training day has null values skip
                if np.any(X.iloc[-1].isnull()):
                    continue

                reconstruction = pd.DataFrame(self.model.inverse_transform(
                    self.model.transform(X.dropna())
                ), columns=self.features)
                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = (
                    np.abs(
                        self.model.named_steps['scaler'].transform(X.dropna()) -
                        self.model.named_steps['scaler'].transform(reconstruction)
                    )
                )[-1]

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10

        re_df['total_re'] = (re_df).sum(axis=1, min_count=1)
        re_df['study_day'] = subject_data['study_day']

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
        kernel: str = 'rbf'
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days
        )
        self.n_components = n_components
        self.model = Pipeline([
                    ('scaler', RobustScaler()),
                    ('svm', OneClassSVM(degree=n_components, kernel=kernel))
                ])
        self.name = 'svm' + '_' + str(n_components)
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
    
    def labelAnomaly(self, subject_data: pd.DataFrame) -> NDArray:
        anomaly_labels = np.full(subject_data.shape[0], 0)

        # Predict if last day of window anomalous
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size:i].dropna(
                    subset=self.features
                )
                if train.shape[0] < (self.window_size - self.max_missing_days):
                    continue

                self.model.fit(train[self.features])

                # Training set + next day
                X = subject_data.iloc[i - self.window_size: i+1][self.features]

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
    ):
        super().__init__(
            features,
            window_size,
            max_missing_days
        )
        self.n_components = n_components
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
        self.name = 'NMF' + '_' + str(n_components)

    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> pd.DataFrame:
        self.validateInputData(subject_data)
        # Sort values
        subject_data = subject_data.sort_values(by='study_day')

        # initialize columns names with features + _re
        df_cols = [
            '{}_re'.format(feature) for feature in self.features
        ]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
            index=subject_data.index
        )
        nmf_components = np.full(
            (
                subject_data.shape[0], len(df_cols), self.n_components
            ),
            np.nan
        )

        # Calculate reconstruction error for each day
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > (self.window_size):
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size:i].dropna(
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
                    batch_size=round(self.window_size/2),
                )
                W = model.train_dict()
                nmf_components[i, :, :] = W

                # Training set + next day reconstructed
                X = scaler.transform(subject_data.iloc[i - self.window_size: i+1][self.features].dropna()).T
                H = model.sparse_code(X, W)
                reconstruction = W @ H

                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = (
                    np.abs(
                        X - reconstruction
                    ).T
                )[-1]

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10

        re_df['total_re'] = (re_df).sum(axis=1, min_count=1)
        re_df['study_day'] = subject_data['study_day']

        # Store pca components in detector class
        self.components: ArrayLike = nmf_components
        self.reconstruction_error = re_df

        return re_df

# Calculate accuracy, sensitivity and specificity
def performance_metrics(
    data: pd.DataFrame,
    anomaly_detector_cols: List[str],
    groupby_cols: List[str] = ['subject_id', 'history_type', 'window_size', 'anomaly_freq'],
) -> pd.DataFrame:
    models = [c.split('_anomaly')[0] for c in anomaly_detector_cols]
    performance_dict = {
        c: [] for c in groupby_cols 
    }
    performance_dict['model'] = []
    performance_dict['true_positives'] = []
    performance_dict['true_negatives'] = []
    performance_dict['false_positives'] = []
    performance_dict['false_negatives'] = []

    for info, subject_data in data.groupby(groupby_cols):
        # Fix error if only one groupby item, info is a string, not a tuple[str]
        if type(info) == str:
            info = [info]
        for model in models:
            for i, val in enumerate(info):
                performance_dict[groupby_cols[i]].append(val)
            performance_dict['model'].append(model)
            performance_dict['true_positives'].append((subject_data['anomaly'] & subject_data[model+'_anomaly']).sum())
            performance_dict['true_negatives'].append((~subject_data['anomaly'] & ~subject_data[model+'_anomaly']).sum())
            performance_dict['false_negatives'].append((subject_data['anomaly'] & ~subject_data[model+'_anomaly']).sum())
            performance_dict['false_positives'].append((~subject_data['anomaly'] & subject_data[model+'_anomaly']).sum())

    performance_df = pd.DataFrame(performance_dict)
    performance_df['sensitivity'] = performance_df['true_positives'] \
        / (performance_df['true_positives'] + performance_df['false_negatives'])
    performance_df['specificity'] = performance_df['true_negatives'] \
        / (performance_df['true_negatives'] + performance_df['false_positives']) 
    performance_df['accuracy'] = performance_df[['true_positives', 'true_negatives']].sum(axis=1) \
        / performance_df[['true_positives', 'true_negatives', 'false_positives', 'false_negatives']].sum(axis=1)
    return performance_df

# Find distance of induced anomaly to closest detected anomaly
# TODO: Test this function
def distance_real_to_detected_anomaly(
    data: pd.DataFrame,
    groupby_cols: List[str],
    anomaly_detector_cols: List[str]
) -> pd.DataFrame:
    anomaly_detector_distances = []

    models = [c.split('_anomaly')[0] for c in anomaly_detector_cols]
    for info, subject_data in data.groupby(groupby_cols):
        # day of detected anomaly
        anomaly_days = {
            c: subject_data.loc[subject_data[c + '_anomaly'], 'study_day'].values
            for c in models 
        }

        # Actual induced anomalies
        real_anomaly = subject_data.loc[
            subject_data['anomaly'],
            'study_day'
        ].values

        # Initialize with NaN values 
        anomaly_distance = {
            c: np.full(real_anomaly.shape, np.nan)
            for c in models 
        }

        # Calculate distance for each induced anomaly to closest future detected anomaly
        for i in range(real_anomaly.shape[0]):
            for c in models:
                distances = anomaly_days[c] - real_anomaly[i] 
                pos_distances = distances[distances >= 0]

                # If no future detected anomaly, fill distance with np.nan
                if np.any(pos_distances):
                    min_pos = np.min(pos_distances)
                else:
                    min_pos = np.nan
                anomaly_distance[c][i] = min_pos

        anomaly_distance = pd.DataFrame(anomaly_distance)
        for i, c in enumerate(groupby_cols):
            anomaly_distance[c] = info[i]
        anomaly_detector_distances.append(anomaly_distance)

    # for each detector's anomalies, find closest day of real anomaly on or after detected, and calculate distance
    # If no real anomaly before detected anomaly, distance = np.nan
    anomaly_detector_distance_df = pd.concat(anomaly_detector_distances)
    anomaly_detector_distance_df = anomaly_detector_distance_df.melt(
        id_vars=groupby_cols,
        var_name='model',
        value_name='distance'
    )
    return anomaly_detector_distance_df

if __name__ == '__main__':
    import load_refs as lr
    import simulate_daily as sd
    all_feature_params = lr.get_all_feature_params()

    for sim_type, feature_params in all_feature_params.items():
        # Load data
        if 'Anomaly' in sim_type:
            simulator = sd.RandomAnomalySimulator(
                feature_params=feature_params,
            )
            dataset = simulator.simulateData()

            for subject, subject_data in dataset.groupby('subject_id'):
                # Detect anomalies
                anomalyDetector = NMFRollingAnomalyDetector(
                    features=feature_params.keys()
                )

                re = anomalyDetector.getReconstructionError(subject_data)
                re['anomaly'] = anomalyDetector.labelAnomaly(re)
                print(re)
                break