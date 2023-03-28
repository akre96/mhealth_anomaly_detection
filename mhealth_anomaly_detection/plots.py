import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import seaborn as sns
from typing import List, Dict, Tuple
import warnings
from pathlib import Path

import mhealth_anomaly_detection.format_axis as fa
import mhealth_anomaly_detection.load_refs as lr


def lineplot_features(
    data: pd.DataFrame,
    plot_features: List,
    time_col: str = "study_day",
    palette: Dict = None,
    width: int = 10,
    anomaly_col: str = None,
    scatter: bool = False,
    hue: str = None,
) -> Tuple[Figure, Axes]:
    if palette is None:
        palette = lr.get_colors()

    fig, axes = plt.subplots(
        figsize=(width, 1.25 * len(plot_features)),
        nrows=len(plot_features),
        ncols=1,
        sharex=True,
        constrained_layout=True,
    )

    for i, f in enumerate(plot_features):
        ax = axes[i]
        if f not in palette["features"]:
            warnings.warn(
                "Color for "
                + f
                + " not in ref/colors.json under features, please add. Defaulting to gray"
            )
            color = "gray"
        else:
            color = palette["features"][f]

        data.loc[f] = pd.to_numeric(data[f], errors="coerce")
        data.loc[time_col] = pd.to_numeric(data[time_col], errors="coerce")
        kwargs = {
            "x": data.dropna(subset=[f])[time_col],
            "y": data.dropna(subset=[f])[f].astype(float),
            "color": color,
            "ax": ax,
        }

        if hue is not None:
            kwargs["hue"] = data[hue]
            kwargs.pop("color")

        # Weird matplotlib error around datatypes
        try:
            sns.lineplot(**kwargs)
        except TypeError:
            pass

        if scatter:
            sns.scatterplot(**kwargs)
        if f == "ema_sad_choices":
            ax.set_ylim([-0.1, 3.1])
            ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel("")
        ax.set_title(f, fontsize=20, color=color)

        if (i > 0) and (hue is not None):
            ax.legend().remove()

        # Label days with anomalies detected
        if anomaly_col is not None:
            anomaly_days = data.loc[data[anomaly_col], "study_day"]
            ylims = ax.get_ylim()
            ax.vlines(anomaly_days, *ylims, lw=2, color="red", alpha=0.3)
            ax.set_ylim(*ylims)
        fa.despine_thicken_axes(ax, fontsize=20, lw=4)
    return fig, axes


def overlay_reconstruction_error(
    reconstruction_error: pd.DataFrame,
    fig: Figure,
    axes: Axes,
    plot_features: List,
    palette: Dict,
    time_col: str = "study_day",
) -> None:
    for i, f in enumerate(plot_features):
        re_f = f + "_re"
        if re_f not in reconstruction_error.columns:
            continue

        ax = axes[i].twinx()

        if f not in palette["features"]:
            color = "gray"
        else:
            color = palette["features"][f]
        try:
            sns.lineplot(
                y=re_f,
                x=time_col,
                ax=ax,
                color=color,
                linestyle="dashed",
                alpha=0.5,
                data=reconstruction_error,
            )
        except TypeError:
            sns.despine(ax=ax)
            continue


if __name__ == "__main__":
    import mhealth_anomaly_detection.anomaly_detection as ad

    print('Plotting all subjects from cached simulations in "cache/*.csv"')
    all_feature_params = lr.get_all_feature_params()
    files = Path("cache").glob("*.csv")
    palette = lr.get_colors()

    for f in files:
        # Get simulation parameters saved in file name
        file_params_str = f.name.split(".")[0].split("_")
        if f.name.startswith("exp"):
            continue
        file_params = {
            val.split("-")[0]: val.split("-")[1] for val in file_params_str[1:]
        }
        file_params["sim_type"] = f.name.split("_")[0]

        if file_params["sim_type"] not in all_feature_params.keys():
            print("Skipping", f)
            continue

        # Initialize correct anomaly detector
        detectors = {
            "base": ad.BaseRollingAnomalyDetector(
                all_feature_params[file_params["sim_type"]].keys()
            ),
            "pca": ad.PCARollingAnomalyDetector(
                all_feature_params[file_params["sim_type"]].keys()
            ),
            "nmf": ad.NMFRollingAnomalyDetector(
                all_feature_params[file_params["sim_type"]].keys()
            ),
            "svm": ad.SVMRollingAnomalyDetector(
                all_feature_params[file_params["sim_type"]].keys()
            ),
        }

        for detector_name, anomalyDetector in detectors.items():
            # Create directory to save images
            fig_dir = Path("output", "plot_simulation", detector_name)
            if not fig_dir.exists():
                fig_dir.mkdir(parents=True, exist_ok=True)

            # Load simulation data and plot each participant
            data = pd.read_csv(f)
            for sid, subject_data in data.groupby("subject_id"):
                subject_data = subject_data.reset_index()
                subject_data["anomaly"] = anomalyDetector.labelAnomaly(
                    subject_data
                )
                subject_data["total_re"] = np.nan
                if not anomalyDetector.reconstruction_error.empty:
                    re = anomalyDetector.getReconstructionError(subject_data)
                    subject_data["total_re"] = re["total_re"]

                fig, axes = lineplot_features(
                    subject_data,
                    [
                        "total_re",
                        "ema_sad_choices",
                        *all_feature_params[file_params["sim_type"]].keys(),
                    ],
                    anomaly_col="anomaly",
                )
                if not anomalyDetector.reconstruction_error.empty:
                    overlay_reconstruction_error(
                        re,
                        fig,
                        axes,
                        [
                            "total_re",
                            "ema_sad_choices",
                            *all_feature_params[
                                file_params["sim_type"]
                            ].keys(),
                        ],
                        palette=palette,
                    )
                plt.suptitle(
                    sid
                    + " - "
                    + " - ".join(
                        [
                            k + ": " + v
                            for k, v in file_params.items()
                            if k not in ["nSubject"]
                        ]
                    )
                )
                fname = Path(
                    fig_dir,
                    sid
                    + "_"
                    + "_".join(
                        [
                            k + "-" + v
                            for k, v in file_params.items()
                            if k not in ["nSubject"]
                        ]
                    )
                    + ".png",
                )
                print(fname)
                fig.savefig(fname)
