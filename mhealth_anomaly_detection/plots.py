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
) -> Tuple[Figure, Axes]:
    if palette is None:
        palette = lr.get_colors()

    fig, axes = plt.subplots(
        figsize=(width, 2*len(plot_features)),
        nrows=len(plot_features),
        ncols=1,
        sharex=True
    )

    plt.subplots_adjust(hspace=.6)
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
        sns.lineplot(
            x=time_col,
            y=f,
            color=color,
            ax=ax,
            data=data,
        )
        if f == 'ema_sad_choices':
            ax.set_ylim([-.1, 3.1])
            ax.set_yticks([0, 1, 2, 3])
        ax.set_ylabel('')
        ax.set_title(f, fontsize=20, color=color)
        fa.despine_thicken_axes(ax, fontsize=20, lw=6)
    return fig, axes


if __name__ == '__main__':
    print('Plotting all subjects from cached simulations in "cache/*.csv"')
    feature_params = lr.get_feature_params()
    files = Path('cache').glob('*.csv')
    for f in files:
        # Get simulation parameters saved in file name
        file_params_str = f.name.split('.')[0].split('_')
        file_params = {
            val.split('-')[0]: val.split('-')[1] for val in file_params_str[1:]
        }
        file_params['sim_type'] = f.name.split('_')[0]

        # Create directory to save images
        fig_dir = Path('output', 'plot_simulation', file_params['sim_type'])
        if not fig_dir.exists():
            fig_dir.mkdir(
                parents=True,
                exist_ok=True
            )

        # Load simulation data and plot each participant
        data = pd.read_csv(f)
        for sid, subject_data in data.groupby('subject_id'):
            fig, axes = lineplot_features(
                subject_data,
                [
                    'ema_sad_choices',
                    *feature_params[file_params['sim_type']].keys(),
                ]
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
                        if k not in ['nSubject', 'sim_type']
                    ]
                ) + '.png'

            )
            print(fname)
            fig.savefig(fname)