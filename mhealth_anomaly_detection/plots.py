import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import seaborn as sns
from typing import List, Dict
import warnings

import mhealth_anomaly_detection.format_axis as fa
import mhealth_anomaly_detection.load_refs as lr

def lineplot_features(
    data: pd.DataFrame,
    plot_features: List,
    time_col: str = 'study_day',
    palette: Dict = None,
    width: int = 10,
) -> Axes:
    if palette is None:
        palette = lr.get_colors()

    _, axes = plt.subplots(
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
    return axes


