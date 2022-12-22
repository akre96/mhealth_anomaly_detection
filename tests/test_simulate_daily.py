# TODO: test that simulation using base class and anomaly class with same parameters gives same results
# TODO: test that simulation using base class and anomaly class with different parameters gives different results
# TODO: test anomalies input at desired frequency
from mhealth_anomaly_detection import simulate_daily as sd


def test_base_case_single():
    simulator = sd.BaseDailyDataSimulator(
        feature_params={
            'example_feature': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
            }
        },
        n_days=1,
        n_subjects=1,
        sim_type='base',
        cache_simulation=False,
    )
    data = simulator.simulateData()
    print(data)
    assert data.subject_id.nunique() == 1
    assert data.study_day.nunique() == 1
    assert 'example_feature' in data.columns
    assert data.shape == (1, 4)

def test_base_case_several():
    n_subs = 5
    n_days = 5
    n_feat = 2
    simulator = sd.BaseDailyDataSimulator(
        feature_params={
            'example_feature': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
            },
            'example_feature_2': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
            }
        },
        n_days=n_days,
        n_subjects=n_subs,
        sim_type='base',
        cache_simulation=False,
    )
    data = simulator.simulateData()
    print(data)
    assert data.subject_id.nunique() == n_subs
    assert data.study_day.nunique() == n_days
    assert 'example_feature_2' in data.columns
    assert data.shape == (n_days*n_subs, 3 + n_feat)

def test_anomaly_case_single():
    simulator = sd.RandomAnomalySimulator(
        feature_params={
            'example_feature': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
                'anomaly_frequency': 7,
                'anomaly_std_scale': 2,
            }
        },
        n_days=1,
        n_subjects=1,
        sim_type='base',
        cache_simulation=False,
    )
    data = simulator.simulateData()
    print(data)
    assert data.subject_id.nunique() ==1
    assert data.study_day.nunique() == 1
    assert 'example_feature' in data.columns
    assert data.shape == (1, 4)

def test_base_case_several():
    n_subs = 5
    n_days = 5
    n_feat = 2
    simulator = sd.RandomAnomalySimulator(
        feature_params={
            'example_feature': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
                'anomaly_frequency': 7,
                'anomaly_std_scale': 2,
            },
            'example_feature_2': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
                'anomaly_frequency': 7,
                'anomaly_std_scale': 2,
            }
        },
        n_days=n_days,
        n_subjects=n_subs,
        sim_type='base',
        cache_simulation=False,
    )
    data = simulator.simulateData()
    print(data)
    assert data.subject_id.nunique() == n_subs
    assert data.study_day.nunique() == n_days
    assert 'example_feature_2' in data.columns
    assert data.shape == (n_days*n_subs, 3 + n_feat)

def test_base_case_several_history():
    n_subs = 5
    n_days = 5
    n_feat = 2
    simulator = sd.RandomAnomalySimulator(
        feature_params={
            'example_feature': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 3,
                'anomaly_frequency': 7,
                'anomaly_std_scale': 2,
            },
            'example_feature_2': {
                'min': 0,
                'max': 1,
                'mean': .5,
                'std': 1,
                'history_len': 1,
                'anomaly_frequency': 7,
                'anomaly_std_scale': 2,
            }
        },
        n_days=n_days,
        n_subjects=n_subs,
        sim_type='base',
        cache_simulation=False,
    )
    data = simulator.simulateData()
    print(data)
    assert data.subject_id.nunique() == n_subs
    assert data.study_day.nunique() == n_days
    assert 'example_feature_2' in data.columns
    assert data.shape == (n_days*n_subs, 3 + n_feat)