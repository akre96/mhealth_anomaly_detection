import pandas as pd
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
    time_col: str = 'study_day',
    palette: Dict = None,
    width: int = 10,
    anomaly_col: str = None,
    scatter: bool = False,
    hue: str = None
) -> Tuple[Figure, Axes]:
    if palette is None:
        palette = lr.get_colors()

    fig, axes = plt.subplots(
        figsize=(width, 1.25*len(plot_features)),
        nrows=len(plot_features),
        ncols=1,
        sharex=True,
        constrained_layout=True,
    )

    for i, f in enumerate(plot_features):
        ax = axes[i]
        if f not in palette['features']:
            warnings.warn(
                "Color for " +
                f +
                " not in ref/colors.json under features, please add. Defaulting to gray"
            )
            color = 'gray'
        else:
            color = palette['features'][f]
        kwargs = {
            'x': time_col,
            'y': f,
            'color': color,
            'ax': ax,
            'data': data,
        }
        if hue is not None:
            kwargs['hue'] = hue
            kwargs.pop('color')
        sns.lineplot(**kwargs)
        if scatter:
            sns.scatterplot(**kwargs)
        if f == 'ema_sad_choices':
            ax.set_ylim([-.1, 3.1])
            ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel('')
        ax.set_title(f, fontsize=20, color=color)

        if (i > 0) and (hue is not None):
            ax.legend().remove()

        # Label days with anomalies detected
        if anomaly_col is not None:
            anomaly_days = data.loc[data[anomaly_col], 'study_day']
            ylims = ax.get_ylim()
            ax.vlines(anomaly_days, *ylims, lw=4, color='red', alpha=.5)
            ax.set_ylim(*ylims)
        fa.despine_thicken_axes(ax, fontsize=20, lw=4)
    return fig, axes


if __name__ == '__main__':
    import mhealth_anomaly_detection.anomaly_detection as ad
    print('Plotting all subjects from cached simulations in "cache/*.csv"')
    all_feature_params = lr.get_all_feature_params()
    files = Path('cache').glob('*.csv')

    for f in files:
        # Get simulation parameters saved in file name
        file_params_str = f.name.split('.')[0].split('_')
        file_params = {
            val.split('-')[0]: val.split('-')[1] for val in file_params_str[1:]
        }
        file_params['sim_type'] = f.name.split('_')[0]

        # Initialize correct anomaly detector
        anomalyDetector = ad.BaseRollingAnomalyDetector(
            all_feature_params[file_params['sim_type']].keys(),
        )

        # Create directory to save images
        fig_dir = Path('output', 'plot_simulation', 'pca_3_ad')
        if not fig_dir.exists():
            fig_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        # Load simulation data and plot each participant
        data = pd.read_csv(f)
        for sid, subject_data in data.groupby('subject_id'):
            subject_data = subject_data.reset_index()
            re, components = anomalyDetector.getReconstructionError(subject_data)
            subject_data['anomaly'] = anomalyDetector.labelAnomaly(re)
            subject_data['total_re'] = re['total_re']
            fig, axes = lineplot_features(
                subject_data,
                [
                    'total_re',
                    'ema_sad_choices',
                    *all_feature_params[file_params['sim_type']].keys(),
                ],
                anomaly_col='anomaly',
            )
            plt.suptitle(
                sid + ' - ' + ' - '.join(
                    [
                        k + ': ' + v for k, v in file_params.items()
                        if k not in ['nSubject']
                    ]
                )
            )
            fname = Path(
                fig_dir,
                sid + '_' + '_'.join(
                    [
                        k + '-' + v for k, v in file_params.items()
                        if k not in ['nSubject']
                    ]
                ) + '.png'

            )
            print(fname)
            fig.savefig(fname)