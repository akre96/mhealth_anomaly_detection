import mhealth_anomaly_detection.anomaly_detection as ad
import pandas as pd
import numpy as np
# TODO: Test that base detector and randomAnomaly have same output if 'base' feature params passed

N_DAYS = 10
N_FEATURES = 3
FEATURES = {
    f'example_feature_{i}': np.arange(N_DAYS)*i for i in range(N_FEATURES)
}
TEST_DATA = pd.DataFrame({
    'subject_id': ['test_1'] * N_DAYS,
    'study_day': range(N_DAYS),
    **FEATURES
})


# Test that base detector outputs expected type and shape
def test_input_output_base():
    test_data = TEST_DATA
    detector = ad.BaseRollingAnomalyDetector(
        features=FEATURES.keys(),
        window_size=3,
    )
    print(test_data)
    re_df, pca_components = detector.getReconstructionError(test_data)

    # in re_df expect 1 column + 1 column per feature
    assert re_df.shape == (N_DAYS, N_FEATURES + 1)

    # Captured pca component expected shape
    assert pca_components.shape == (
        N_DAYS,
        detector.model.named_steps['pca'].n_components_,
        N_FEATURES
    )

# Input data cannot be missing full row for a day
def test_input_output_base_missing_day():
    # drop a day of data
    test_data = TEST_DATA.drop(2)

    error = False
    try:
        detector = ad.BaseRollingAnomalyDetector(
            features=list(FEATURES.keys()),
            window_size=3,
        )
        detector.getReconstructionError(test_data)
    except ValueError:
        error = True

    assert error

def test_input_output_base_na_value():
    # drop a day of data
    test_data = TEST_DATA.copy()

    # Set 1 value to NaN
    test_data[list(FEATURES.keys())[1]].iloc[2] = np.nan
    detector = ad.BaseRollingAnomalyDetector(
        features=list(FEATURES.keys()),
        window_size=3,
        max_missing_days=0
    )
    print(test_data)
    re_df, pca_components = detector.getReconstructionError(test_data)

    # in re_df expect 1 column + 1 column per feature
    assert re_df.shape == (N_DAYS, N_FEATURES + 1)

    # Captured pca component expected shape
    assert pca_components.shape == (
        N_DAYS,
        detector.model.named_steps['pca'].n_components_,
        N_FEATURES
    )