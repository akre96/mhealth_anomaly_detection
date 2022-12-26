# Anomaly Detection for mHealth Data


## Files
- `mhealth_anomaly_detection/`: functions used to simulate and do anomaly detection on mHealth dataset
    - `anomaly_detection.py`: Classes for rolling window based anomaly detection
    - `datasets.py`: Loads and pre-processes external datasets, currently loads the CrossCheck daily dataset
    - `format_axis.py`: Helper function to format plot axes
    - `load_refs.py`: Helper functions to load static files
    - `plots.py`: Functions used to generate figures
    - `simulate_daily.py`: Simulator of daily mHealth data
- `tests/`: Unit tests for functions
- `cache/`: holds simulated anomaly data
- `lib/`: holds static referenced files
    - `colors.json`: Color palettes used for different plots
    - `feature_parameters.json`: Statistical parameters for simulated features
- `output/`: Contains plots and results
- `run_scripts/`: Scripts running and combining functions within mHealth_anomaly_detection folder to generate results
- `notebooks/`: Jupyter Notebooks for example analyses
- `poetry.lock`: Lock file used by poetry for python environment
- `pyproject.toml`: Packages and requirements for project



