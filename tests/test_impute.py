import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from hypothesis import given, strategies as st
import pytest

from mhealth_anomaly_detection import impute


@pytest.mark.parametrize("skip_nan", [True, False])
@given(
    n_days=st.integers(min_value=1, max_value=100),
    n_features=st.integers(min_value=1, max_value=50),
    missing_data=st.floats(min_value=0, max_value=1),
)
def test_rolling_impute(n_days, n_features, missing_data, skip_nan):
    missing_days = int(round(n_days) * missing_data)
    data_days = n_days - missing_days
    data = pd.DataFrame(
        {
            "subject_id": ["test"] * n_days,
            "study_day": list(range(n_days)),
            **{
                f"feature_{i}": [1] * data_days + [np.nan] * missing_days
                for i in range(n_features)
            },
        }
    )
    features = [f"feature_{i}" for i in range(n_features)]
    imputer = SimpleImputer(strategy="median")
    filled = impute.rollingImpute(
        data, features, 0, imputer, 1, skip_all_nan=skip_nan
    )
    for i in range(n_features):
        if missing_data == 1.0:
            assert filled[f"feature_{i}"].isna().sum() == n_days
        elif skip_nan:
            assert filled[f"feature_{i}"].isna().sum() == missing_days
        else:
            assert filled[f"feature_{i}"].isna().sum() == 0
            assert (filled[f"feature_{i}"] == 1).all()
