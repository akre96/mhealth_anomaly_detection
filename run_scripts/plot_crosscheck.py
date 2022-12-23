# Plot pca based anomaly detection for a select number of participants in cross check study
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from mhealth_anomaly_detection import \
    datasets, anomaly_detection, plots, load_refs

if __name__ == '__main__':
    # Load data
    crosscheck = datasets.CrossCheck()
    data = crosscheck.data

    # Subjects of interest for plotting
    plot_subjects = [
        118,  # high data availability
        109,  # Variable EMA responses
        95,   # Variable EMA responses
        97,   # Highly variable EMA responses
        84,   # Drop in EMA positive score
    ]

    # Where to save plots
    fig_dir = Path('output', 'crosscheck', 'lineplot')
    fig_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Selection of features to use for anomaly detection
    daily_passive_features = [
        'sleep_duration',
        'act_in_vehicle_ep_0',
        'act_on_bike_ep_0',
        'act_on_foot_ep_0',
        'act_still_ep_0',
        'act_tilting_ep_0',
        'act_unknown_ep_0',
        'audio_amp_mean_ep_0',
        'audio_convo_duration_ep_0',
        'audio_convo_num_ep_0',
        'audio_voice_ep_0',
        'call_in_duration_ep_0',
        'call_in_num_ep_0',
        'call_out_duration_ep_0',
        'call_out_num_ep_0',
        'light_mean_ep_0',
        'light_mean_ep_1',
        'light_mean_ep_4',
        'loc_dist_ep_0',
        'loc_visit_num_ep_0',
        'sms_in_num_ep_0',
        'sms_out_num_ep_0',
        'unlock_duration_ep_0',
        'unlock_num_ep_0'
    ]
    
    # Features to also plot including self-reports
    plot_features = [
        'total_re',
        'ema_neg_score',
        'ema_VOICES',
        'ema_SEEING_THINGS',
        'ema_STRESSED',
        'ema_DEPRESSED',
        'ema_HARM',
        'ema_pos_score',
        'ema_HOPEFUL',
        'ema_CALM',
        'ema_SLEEPING',
        'ema_SOCIAL',
        'ema_THINK',
        'quality_activity',
        'quality_audio',
        'quality_gps_on',
        'quality_light',
        'quality_loc',
        *daily_passive_features
    ]
    palette = load_refs.get_colors()

    # AD parameters
    ad_params = [
        {
            'window_size': 30,
            'max_missing_days': 10,
            'n_components': 7
        },
        {
            'window_size': 15,
            'max_missing_days': 3,
            'n_components': 5
        },
        {
            'window_size': 60,
            'max_missing_days': 20,
            'n_components': 10
        },
    ]


    for params in ad_params:
        print('AD with parameters: ', params)
        # PCA based anomaly detection
        detector = anomaly_detection.PCARollingAnomalyDetector(
            features=daily_passive_features,
            **params
        )
        for sid in tqdm(plot_subjects, desc='Anomaly detection plotting'):
            # Get 1 subject of data
            subject_data = data[data.subject_id == sid].copy()
            if subject_data.empty:
                print(sid, 'empty, skipping')
                continue

            # Calculate reconstruction error
            reconstruction_error = detector.getReconstructionError(subject_data)
            subject_data['total_re'] = reconstruction_error['total_re']

            # Label anomalous days
            subject_data['anomaly'] = detector.labelAnomaly(reconstruction_error)

            # Plot
            fig, axes = plots.lineplot_features(
                subject_data,
                plot_features,
                palette=palette,
                anomaly_col='anomaly',
                scatter=True,
            )
            plots.overlay_reconstruction_error(
                reconstruction_error=reconstruction_error,
                fig=fig,
                axes=axes,
                plot_features=plot_features,
                palette=palette
            )
            fig.suptitle(
                f'Subject {sid}. PCA-AD window {params["window_size"]}, max_missing {params["max_missing_days"]}, n_components {params["n_components"]}'
            )

            # Save plot
            fname = Path(
                fig_dir,
                f'{sid}_PCA-AD_components-{params["n_components"]}_window-{params["window_size"]}_maxMissing-{params["max_missing_days"]}_lineplot.png'.replace(' ', '')
            )
            fig.savefig(str(fname))
            plt.close()
