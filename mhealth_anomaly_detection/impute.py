from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from p_tqdm import p_map


def rollingImpute(data, features, min_days, imputer, num_cpus) -> pd.DataFrame:
    def impute(input) -> pd.DataFrame:
        _, data = input
        filled_data = np.full(
                (
                    data.shape[0], len(features)
                ),
                np.nan
        )
        for i in range(data.shape[0]):
            if i > min_days:
                # Training set
                X = data.iloc[0: i][features].values
                X_fill = imputer.fit_transform(X)
                X_fill[~np.isnan(X)] = X[~np.isnan(X)]
                filled_data[i] = X_fill[-1]
        data[features] = filled_data
        return data
    return pd.concat(
        p_map(
            impute,
            data.groupby('subject_id'),
            num_cpus=num_cpus
        )
    )
