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
    ) -> pd.DataFrame:
        # TODO: empty df with null values to place RE in to
        # TODO: initialize columns names with features + _re
        # TODO: return dataframe

        for i in range(subject_data.shape[0]):
            if i > self.window_size:
                train = subject_data.iloc[i - self.window_size:i]
                self.model.fit(train[self.features])

                # Training set + next day
                X = subject_data.iloc[i - self.window_size: i+1][self.features]
                reconstruction = self.model.inverse_transform(
                    self.model.transform(X)
                )
                reconstruction_error = (X - reconstruction)**2
                reconstruction_error_sum = reconstruction_error.sum(axis=1)