""" For exp 4,5,6,8 -- plot results combining window sizes
"""
import sys

# Make imports work
# TODO: Remove this dependency -- worked fine when using poetry, but not just python3
sys.path.insert(0, "/Users/sakre/Code/dgc/mhealth_anomaly_detection")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from colour import Color
from itertools import product
from p_tqdm import p_map
from mhealth_anomaly_detection import datasets
from mhealth_anomaly_detection import anomaly_detection
from mhealth_anomaly_detection import format_axis as fa
from mhealth_anomaly_detection import load_refs as lr
from mhealth_anomaly_detection import plots

YEARS = [2, 3]
EXPERIMENTS = ["exp04", "exp05", "exp06", "exp08"]
GLOBEM = datasets.GLOBEM(load_data=False, data_path="~/")
CACHE_DIR = "cache"
PARALLEL = False

CHECK_MISSING_FEATURES = [
    "f_loc:phone_locations_doryab_locationentropy:allday",
    "f_loc:phone_locations_barnett_circdnrtn:allday",
    "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
    "f_steps:fitbit_steps_intraday_rapids_sumdurationactivebout:allday",
    "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday",
    "f_slp:fitbit_sleep_intraday_rapids_countepisodeasleepunifiedmain:allday",
    "f_slp:fitbit_sleep_summary_rapids_firstbedtimemain:allday",
    "f_slp:fitbit_sleep_summary_rapids_avgefficiencymain:allday",
    "f_call:phone_calls_rapids_missed_count:allday",
    "f_call:phone_calls_rapids_incoming_count:allday",
    "f_call:phone_calls_rapids_outgoing_count:allday",
    "f_call:phone_calls_rapids_outgoing_sumduration:allday",
]


def run_plots(inputs) -> None:
    year, experiment = inputs
    out_dir = Path("output", f"GLOBEM_year-{year}", experiment)
    print(f"Loading cached data from {experiment} year {year}")
    try:
        exp_ad = pd.read_csv(
            Path(CACHE_DIR, f"GLOBEM-{year}_{experiment}_intermediate.csv"),
            low_memory=False,
        )
        exp = pd.read_csv(Path(CACHE_DIR, f"GLOBEM-{year}_{experiment}.csv"))
    except FileNotFoundError:
        print(
            f"\tIntermediate files for GLOBEM Year {year} {experiment} do not exist in {CACHE_DIR}"
        )
        return
    p_list = []

    print("\tCombining rolling windows in PHQ periods")
    features = [c for c in CHECK_MISSING_FEATURES if c in exp_ad.columns]
    print(
        f"\tUsing {len(features)} of {len(CHECK_MISSING_FEATURES)} features for checking missing days"
    )
    for period in [1, 2, 3]:
        p_list.append(GLOBEM.get_phq_periods(exp_ad, features=features, period=period))
    exp_comb = pd.concat(p_list)
    anomaly_detector_cols = [d for d in exp_ad.columns if d.endswith("_anomaly")]

    print(f"\n\tPlotting to {out_dir}...")
    if not out_dir.exists():
        out_dir.mkdir(parents=True)

    palette = lr.get_colors()

    print("\t\tPlotting anomalies detected per window size")
    anoms = (
        exp_ad.groupby(["subject_id", "window_size"])[anomaly_detector_cols]
        .sum()
        .reset_index()
    )
    n_detectors = len(anomaly_detector_cols)
    nr = 2
    nc = 5
    fs = (20, 10)
    if n_detectors > 10:
        nr = 3
        nc = 4
        fs = (15, 15)
    fig, axes = plt.subplots(figsize=fs, nrows=nr, ncols=nc)
    gray = Color("black")

    for i, ad in enumerate(anomaly_detector_cols):
        ax = axes.flatten()[i]
        dname = ad.split("_anomaly")[0]
        base_c = Color(palette["model"][dname])
        p = list(base_c.range_to(gray, 7))
        for i, (ws, w_df) in enumerate(anoms.groupby("window_size")):
            sns.histplot(
                w_df[ad], color=p[i].hex_l, ax=ax, bins=10, alpha=0.5, label=ws
            )
        ax.set_title(dname.replace("_", " "), fontsize=20)
        ax.legend()
        ax.set_xlabel("")
        fa.despine_thicken_axes(ax, fontsize=15)
    fname = Path(out_dir, "count_anomalies_detected.png")
    plt.tight_layout()
    plt.gcf().savefig(str(fname))

    print("\t\tPlotting correlation of detector to phq target")
    phq_anomalies_qc = exp_comb[
        (exp_comb.days >= exp_comb.period * 6) & (exp_comb.days <= exp_comb.period * 8)
    ]
    info_cols = [
        "subject_id",
        "period",
    ]
    targets = ["phq_start", "phq_change", "phq_stop"]
    for target in targets:
        corr = anomaly_detection.correlateDetectedToOutcome(
            phq_anomalies_qc,
            anomaly_detector_cols,
            outcome_col=target,
            groupby_cols=info_cols,
        )
        corr["r2"] = corr["rho"] ** 2
        corr["model"] = corr["detector"].str.split("_anomaly").str[0]

        # Indiviudal spearman r performance for 1 model
        if target == "phq_stop":
            fig, ax = plt.subplots(figsize=(3, 30))
            detector = "PCA_005_anomaly"
            subset = corr[(corr.period == 1) & (corr.detector == detector)].sort_values(
                by="rho"
            )
            subset["log10p"] = np.log10(subset["p"])
            subset["sig"] = subset["p"] < 0.05
            sns.barplot(
                x="rho",
                y="subject_id",
                hue="sig",
                data=subset,
                dodge=False,
                ax=ax,
            )
            ax.set_xlim(-1, 1)
            ax.set_title(f"model: {detector}, period: {1}")
            ax.grid(axis="x", alpha=0.5, ls="dashed")
            fname = Path(out_dir, f'individual_spearman{"rho"}_{target}_{detector}.png')
            fa.despine_thicken_axes(ax, fontsize=10)
            plt.tight_layout()
            plt.gcf().savefig(str(fname))
            plt.close()

            # Plot top-2, middle, and bottom-2 performer
            p1_corr = corr[corr.period == 1].sort_values(by="rho")
            inds = [0, 1, 2, 3, round(p1_corr.shape[0] / 2), -3, -2, -1]
            p1_corr.iloc[inds].to_csv(Path(out_dir, f"example_subs.csv"), index=False)
            for ind in inds:
                sid = p1_corr.iloc[ind]["subject_id"]
                ad_col = p1_corr.iloc[ind]["detector"]
                rho = p1_corr.iloc[ind]["rho"]
                window_size = 7
                subject_data = exp_ad[
                    (exp_ad.subject_id == sid) & (exp_ad.window_size == window_size)
                ].copy()
                subject_data[ad_col] = subject_data[ad_col].fillna(False)
                _, axes = plots.lineplot_features(
                    subject_data,
                    plot_features=["phq4"] + features,
                    scatter=True,
                    palette=palette,
                )
                pal = list(Color("red").range_to(Color("black"), 3))
                for i, (ws, w_df) in enumerate(
                    exp_ad[exp_ad.subject_id == sid].groupby("window_size")
                ):
                    ads = w_df[w_df[ad_col] == 1].study_day

                    for ax in axes.flatten():
                        ylims = ax.get_ylim()
                        ax.vlines(ads, *ylims, color=pal[i].hex_l, lw=4, alpha=0.5)
                        ax.set_ylim(*ylims)
                fname = Path(out_dir, f"{sid}_{ad_col}_rho{round(rho, 2)}_top{ind}.png")
                fa.despine_thicken_axes(ax, fontsize=10)
                plt.tight_layout()
                plt.gcf().savefig(str(fname))
                plt.close()

        # Clustermap across participants
        vals = corr.pivot_table(
            index="subject_id",
            columns=["period", "detector"],
            values="rho",
        )

        sns.clustermap(vals, cmap="coolwarm", vmin=-1, vmax=1, figsize=(20, 20))
        fname = Path(out_dir, f'individual_clustermap_spearman{"rho"}_{target}.png')
        plt.tight_layout()
        plt.gcf().savefig(str(fname))
        plt.close()

        for metric in ["rho", "r2"]:
            corr_table = corr.pivot_table(
                index=["detector"], columns=["period"], values=metric, aggfunc="median"
            )
            hm_size = (10, 7)
            fig, ax = plt.subplots(figsize=hm_size)
            sns.heatmap(
                corr_table,
                center=0,
                vmin=-1,
                vmax=1,
                square=True,
                annot=True,
                cmap="coolwarm",
                ax=ax,
            )
            fname = Path(out_dir, f"spearman{metric}_{target}_heatmap.png")
            fa.despine_thicken_axes(ax, heatmap=True, fontsize=12, x_tick_fontsize=10)
            plt.tight_layout()
            plt.gcf().savefig(str(fname))
            plt.close()

            # Plot box-whisker of individual subject-level performance
            model_order = [
                "RollingMean",
                "PCA_003",
                "PCA_005",
                "PCA_010",
                "PCA_020",
                "NMF_003",
                "NMF_005",
                "NMF_010",
                "NMF_020",
                "SVM_rbf",
                "SVM_sigmoid",
                "SVM_poly",
            ]
            model_order = [c for c in model_order if c in corr.model.unique()]
            l = 0.8
            p_int = {m: Color(c) for m, c in palette["model"].items()}
            for m, c in p_int.items():
                c.set_luminance(l)

            p_t = {m: c.get_hex_l() for m, c in p_int.items()}
            fig, axes = plt.subplots(figsize=(10, 5), ncols=2)
            plt.subplots_adjust(wspace=0.6)
            ns = corr[corr.p > 0.05]
            sig = corr[corr.p < 0.05]
            p_transparent = {m: Color(c).get_rgb() for m, c in palette["model"].items()}
            for i, metric in enumerate(["rho", "r2"]):
                ax = axes[i]
                sns.stripplot(
                    y="model",
                    x=metric,
                    data=sig,
                    hue="model",
                    palette=palette["model"],
                    order=model_order,
                    alpha=0.7,
                    ax=ax,
                )
                sns.stripplot(
                    y="model",
                    x=metric,
                    data=ns,
                    hue="model",
                    palette="gray",
                    order=model_order,
                    alpha=0.1,
                    ax=ax,
                )
                sns.boxplot(
                    y="model",
                    x=metric,
                    data=corr,
                    hue="model",
                    palette=p_t,
                    # palette=palette['model'],
                    # color='white',
                    saturation=1,
                    order=model_order,
                    dodge=False,
                    ax=ax,
                )
                fa.despine_thicken_axes(ax, fontsize=15)
                ax.legend().remove()
                ax.set_xlabel("Spearman Rho")
                corr.groupby("model")[["rho", "r2"]].describe().round(3).to_csv(
                    Path(out_dir, f"{target}_corr_model_summary.csv")
                )
            ax.set_ylabel("")
            ax.set_xlabel("R-squared")

            fname = Path(out_dir, f"box-whisker_r_rho_{target}_heatmap.png")
            plt.tight_layout()
            plt.gcf().savefig(str(fname))
            plt.close()
    return


conditions = list(product(YEARS, EXPERIMENTS))
if PARALLEL:
    p_map(run_plots, conditions)
else:
    for inputs in conditions:
        run_plots(inputs)
