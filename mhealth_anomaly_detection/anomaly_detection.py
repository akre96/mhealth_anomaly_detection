import numpy as np
import pandas as pd
from typing import Any, Tuple
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class BaseRollingAnomalyDetector:
    def __init__(
        self,
        features: list,
        window_size: int = 7,
        model: Any = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=3, whiten=True))
        ]),
    ):
        self.window_size = window_size
        self.model = model
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
                subject_data.shape[0], 3, len(df_cols)
            ),
            np.nan
        )
        # Calculate reconstruction error for each day
        for i in range(subject_data.shape[0]):
            # RE only calculated with sufficient historical data
            if i > self.window_size:
                # Train on window_size days
                train = subject_data.iloc[i - self.window_size:i]
                self.model.fit(train[self.features])
                components[i, :, :] = self.model.named_steps['pca'].components_

                # Training set + next day reconstructed
                X = subject_data.iloc[i - self.window_size: i+1][self.features]
                reconstruction = self.model.inverse_transform(
                    self.model.transform(X)
                )
                # Reconstruction error for out-of-training day kept
                re_df.iloc[i] = ((X - reconstruction)**2).iloc[-1]

        re_df['total_re'] = re_df.sum(axis=1)
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