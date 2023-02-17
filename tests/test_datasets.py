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