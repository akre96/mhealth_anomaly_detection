import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


class BaseRollingAnomalyDetector:
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        max_missing_days: int = 2,
        n_components: int = 3,
    ):
        self.n_components = n_components
        self.model = Pipeline([
                    ('scaler', RobustScaler()),
                    ('pca', PCA(n_components=n_components, whiten=True))
                ])
        self.window_size = window_size
        self.max_missing_days = max_missing_days
        self.features = features
    
    def getReconstructionError(
        self,
        subject_data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        # data validation
        if 'subject_id' not in subject_data:
            raise ValueError('Subject data must have column subject_id')
        if subject_data.subject_id.nunique() != 1:
            raise ValueError('Subject data must have only one unique subject_id')

        day_range = subject_data.study_day.max() - subject_data.study_day.min()
        if subject_data.shape[0] != day_range+1:
            raise ValueError('Subject data must have a row for every day even if missing feature values')
        
        # Sort values
        subject_data.sort_values(by='study_day', inplace=True)

        # initialize columns names with features + _re
        df_cols = [
            '{}_re'.format(feature) for feature in self.features
        ]

        # empty df with null values to place RE in to
        re_df = pd.DataFrame(
            data=np.full((subject_data.shape[0], len(df_cols)), np.nan),
            columns=df_cols,
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

                # Training set + next day reconstructed
                X = subject_data.iloc[i - self.window_size: i+1][self.features].dropna()
                reconstruction = pd.DataFrame(self.model.inverse_transform(
                    self.model.transform(X)
                ), columns=self.features)
                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = (
                    (
                        self.model.named_steps['scaler'].transform(X) -
                        self.model.named_steps['scaler'].transform(reconstruction)
                    )**2
                )[-1]

        # Clip reconstruction error to 10
        re_df[re_df > 10] = 10

        re_df['total_re'] = re_df.sum(axis=1, min_count=1)
        re_df['study_day'] = subject_data['study_day']
        return re_df, pca_components

    # Return if anomalous day labels
    def labelAnomaly(self, re_df: pd.DataFrame) -> pd.Series:
        # anomaly as mean + 2*std of reconstruction error
        anomaly_threshold = re_df['total_re'].mean() + 2*re_df['total_re'].std()
        return re_df['total_re'] > anomaly_threshold


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
                anomalyDetector = BaseRollingAnomalyDetector(
                    features=feature_params.keys()
                )

                re, _ = anomalyDetector.getReconstructionError(subject_data)
                re['anomaly'] = anomalyDetector.labelAnomaly(re)
                print(re)
                break