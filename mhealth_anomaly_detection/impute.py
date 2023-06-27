from tqdm.auto import tqdm
import pandas as pd
import numpy as np
from p_tqdm import p_map
from tqdm.auto import tqdm
from typing import List, Any


def rollingImpute(
    data: pd.DataFrame,
    features: List[str],
    min_days: int,
    imputer: Any,
    num_cpus: int,
    skip_all_nan: bool = True,
) -> pd.DataFrame:
    """Imputes missing values in a rolling window fashion per subject ID.

    Args:
        data (pd.DataFrame): data to impute, requires study_day and subject_id columns
        features (List[str]): column names of features
        min_days (int): Minimum days of data before imputing
        imputer (Any): Any imputer that has a fit_transform method
        num_cpus (int): Number of cpus to use, if 1, will not use multiprocessing
        skip_all_nan (bool, optional): If true, skip values where all features are NaN . Defaults to True.

    Returns:
        pd.DataFrame: _description_
    """

    def impute(input) -> pd.DataFrame:
        _, data = input
        filled_data = data[features].copy().to_numpy()
        if data.shape[0] < 2:
            return data
        for i in range(1, data.shape[0]):
            vals = data.iloc[0 : i + 1][features]
            if vals.dropna(subset=features, how="all").shape[0] < min_days:
                continue
            if i > min_days:
                # If all values of the day to impute are NaN, skip
                if skip_all_nan and data.iloc[i][features].isna().all():
                    continue

                # Training set
                X = data.iloc[0 : i + 1][features].to_numpy()

                # If all values are NaN, skip
                if np.all(np.isnan(X)):
                    continue

                # Impute missing values
                X_fill = imputer.fit_transform(X)
                X_fill[~np.isnan(X)] = X[~np.isnan(X)]
                filled_data[i] = X_fill[-1]

        data[features] = filled_data
        return data

    if num_cpus == 1:
        filled = []
        for g, g_df in tqdm(
            data.groupby("subject_id"), total=data.subject_id.nunique()
        ):
            filled.append(impute((g, g_df)))
        return pd.concat(filled)
    return pd.concat(
        p_map(
            impute,
            data.sort_values(by="study_day").groupby("subject_id"),
            num_cpus=num_cpus,
        )
    )
