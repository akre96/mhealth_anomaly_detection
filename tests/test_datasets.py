import numpy as np
import pandas as pd
from mhealth_anomaly_detection.datasets import GLOBEM


def test_GLOBEM_survey():
    dataset = GLOBEM(
        data_path='~/Data/mHealth_external_datasets/GLOBEM',
        year=2,
        sensor_data_types=['wifi'],
    )
    surveys = dataset.get_weekly_phq4()
    print(surveys)
    assert surveys.shape == (2051, 3)

def test_PHQ_periods():
    gap = 5
    ndays = 100
    sim_data = pd.DataFrame({
        'subject_id': ['test'] * ndays,
        'test_anomaly': [0, 1, 1, 1, 1] * int(ndays/gap),
        'phq4': ([1] + [np.nan]*(gap-1)) * int(ndays/gap),
        'feat': [0] * ndays,
        'study_day': range(ndays)
    })
    dataset = GLOBEM(
        data_path='~/Data/mHealth_external_datasets/GLOBEM',
        load_data=False,
        year=2,
        sensor_data_types=['wifi'],
    )
    print(sim_data)
    periods = dataset.get_phq_periods(
        data=sim_data,
        features=['feat'],
        period=1
    )
    print(periods)
    assert periods.iloc[0].start == 0
    assert periods.shape[0] == (int(ndays/gap) - 1)
    assert np.all(periods.test_anomaly == 4)

    sim_data = pd.DataFrame({
        'subject_id': ['test'] * ndays,
        'test_anomaly': [np.nan, np.nan, np.nan, np.nan, np.nan] * int(ndays/gap),
        'phq4': ([1] + [np.nan]*(gap-1)) * int(ndays/gap),
        'feat': [0] * ndays,
        'study_day': range(ndays)
    })
    periods = dataset.get_phq_periods(
        data=sim_data,
        features=['feat'],
        period=1
    )
    print(periods)
    assert np.all(periods.test_anomaly.isna())