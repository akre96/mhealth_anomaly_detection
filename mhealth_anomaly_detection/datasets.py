import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
from tqdm.auto import tqdm

class DatasetBase:
    def __init(
        self,
        data_path: str,
    ):
        if Path(data_path).expanduser().is_file():
            self.data_raw = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f'{data_path} does not exist')
        
        self.data = pd.DataFrame()


class GLOBEM(DatasetBase):
    def __init__(
        self,
        data_path: str = '~/Data/mHealth_external_datasets/GLOBEM',
        year: int = 2,
        sensor_data_types: List = [
            'wifi',
            'steps',
            'sleep',
            'screen',
            'rapids',
            'location',
            'call',
            'bluetooth'
        ],
        load_data: bool = True
    ):
        if year not in [2, 3, 4]:
            raise ValueError('Year for the GLOBEM dataset analysis must be 2, 3, or 4')

        self.possible_sensor_types = [
            'wifi',
            'steps',
            'sleep',
            'screen',
            'rapids',
            'location',
            'call',
            'bluetooth'
        ]
        for t in sensor_data_types:
            if t not in self.possible_sensor_types:
                raise ValueError(f'{t} not a possible sensor type in GLOBEM')

        if not Path(data_path).expanduser().is_dir():
            raise FileNotFoundError(
                f'{data_path} does not exist, set data_path to unzipped\
                     GLOBEM dataset from\
                     https://physionet.org/content/globem/1.0'
            )
        else:
            self.data_path = data_path

        self.sensor_data_types = sensor_data_types
        self.year = year
        self.id_cols = ['pid', 'platform', 'date']
        self.feature_cols = []

        if load_data:
            self.data_raw = self.combine_data()
            self.data = self.preprocess(self.data_raw)

            self.sensor_cols = [f for f in self.feature_cols if f != 'phq4']

    def preprocess(self, data_raw) -> pd.DataFrame:
        data = self.filter_high_missing_cols(data_raw).rename(
            columns={
                'pid': 'subject_id',
            }
        )
        self.id_cols = ['subject_id', 'platform', 'date']
        data = self.filter_redundant_cols(data)
        data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
        data = self.add_study_day(data)
        data = self.fill_empty_days(data)
        data = self.filter_high_missing_participants(data)
        data = self.add_missingness_indicator(data)

        return data[self.id_cols + self.feature_cols]

    def add_study_day(self, data) -> pd.DataFrame:
        # Get first date of participant
        first_days = pd.DataFrame(
            data.groupby('subject_id')['date'].min()
        ).reset_index().rename(columns={'date': 'first_day'})

        # Set study day as relative days in study
        reform_data = data.merge(first_days, validate='m:1')
        reform_data['study_day'] = (
            reform_data['date'] - reform_data['first_day']
        ).dt.days
        self.id_cols = self.id_cols + ['study_day']
        return reform_data

    # TODO: Test function
    def fill_empty_days(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        expected_rows = []
        for (sid), s_df in data.groupby('subject_id'):
            min_d = s_df.study_day.min()
            max_d = s_df.study_day.max()
            n_days = 1 + max_d - min_d
            expected_rows.append(
                pd.DataFrame({
                    'subject_id': [sid] * n_days,
                    'study_day': np.arange(min_d, max_d + 1)
                })
            )

        filled = data.merge(
            pd.concat(expected_rows),
            how='right',
            validate='1:1',
        )

        print(
            'Filling empty study days with NaN values going from',
            data.shape[0],
            'to',
            filled.shape[0],
            '. Added ',
            filled.shape[0] - data.shape[0]
        )
        return filled

    def combine_data(self) -> pd.DataFrame:
        surveys = self.get_weekly_phq4()
        sensors = self.get_daily_sensor_data()
        participants = self.get_participant_info()

        participant_sensors = participants.merge(
            sensors,
            how='inner',
            validate='1:m'
        )

        data = participant_sensors.merge(
            surveys,
            how='outer',
            validate='1:1'
        ).sort_values(by=['pid', 'date'])

        return data

    def get_weekly_phq4(self) -> pd.DataFrame:
        weekly_survey_path = Path(
            self.data_path,
            f'INS-W_{self.year}',
            'SurveyData',
            'dep_weekly.csv'
        )

        return pd.read_csv(
            weekly_survey_path,
            index_col=0
        )[['pid', 'date', 'phq4']].dropna()

    def get_daily_sensor_data(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for i, sensor_type in enumerate(self.sensor_data_types):
            sensor_data_path = Path(
                self.data_path,
                f'INS-W_{self.year}',
                'FeatureData',
                f'{sensor_type}.csv'
            )
            df = pd.read_csv(sensor_data_path, index_col=0)
            if i == 0:
                data = df
            else:
                data = data.merge(
                    df,
                    how='outer',
                    validate='1:1'
                )

        return self.filter_nondaily_cols(data)

    def get_participant_info(self) -> pd.DataFrame:
        info_path = Path(
            self.data_path,
            f'INS-W_{self.year}',
            'ParticipantsInfoData',
            'platform.csv'
        )
        return pd.read_csv(
            info_path,
            index_col=0
        ).reset_index()

    def filter_nondaily_cols(self, data) -> pd.DataFrame:
        cols = [c for c in data.columns if c not in self.id_cols]
        use_segments = [
            'morning',
            'afternoon',
            'evening',
            'night',
            'allday',
        ]
        filtered = [c for c in cols if c.split(':')[-1] in use_segments]
        print(f'Filtering only daily features. Going from {len(cols)} to {len(filtered)} features')
        return data[['pid', 'date'] + filtered]

    def filter_high_missing_cols(self, data) -> pd.DataFrame:
        missing = data.isnull().sum().sort_values().T.reset_index()
        # Filter those more missing than PHQ 4
        thresh = missing[missing['index'] == 'phq4'][0].values[0]
        use_cols = list(missing[missing[0] <= thresh]['index'].values)

        print(f'Filtering columns with high missingness. Going from {missing.shape[0]} to {len(use_cols)} features')
        self.feature_cols = [c for c in use_cols if c not in self.id_cols]
        return data[use_cols]

    def filter_redundant_cols(self, data) -> pd.DataFrame:
        feats = self.feature_cols
        # Remove wake time since sleep time present
        filtered = [c for c in feats if 'awake' not in c]

        # Only look at raw data, not normalized or discretized
        redundant_steps = [
            'maxsumsteps',
            'minsumsteps',
            'avgsumsteps',
            'mediansumsteps',
            'stdsumsteps',
            'maxsteps',
            'minsteps',
            'avgsteps',
            'stdsteps',
        ]
        filtered = [
            c for c in filtered if not (
                len(c) > 5 and (
                    c.split(':')[-2].endswith('_norm') or
                    c.split(':')[-2].endswith('_dis') or
                    c.split(':')[-2].split('_')[-1] in redundant_steps
                )
            )
        ]
        print(f'Filtering redundant features. Going from {len(feats)} to {len(filtered)} features')
        self.feature_cols = filtered
        return data[self.id_cols + filtered]

    def filter_high_missing_participants(self, data) -> pd.DataFrame:
        remove_participants = []
        if self.year == 2:
            remove_participants = [
                'INS-W_314',
                'INS-W_316',
                'INS-W_317',
                'INS-W_322',
                'INS-W_335',
                'INS-W_361',
                'INS-W_392',
                'INS-W_394',
                'INS-W_421',
                'INS-W_436',
                'INS-W_460',
                'INS-W_479',
                'INS-W_485',
                'INS-W_493',
                'INS-W_495',
                'INS-W_497',
                'INS-W_505',
                'INS-W_512',
                'INS-W_523',
                'INS-W_527',
                'INS-W_536',
                'INS-W_537',
                'INS-W_555',
                'INS-W_556',
                'INS-W_569',
                'INS-W_570',
            ]
        else:
            print('Warning: Filtering participants not established for year 1,3,4')

        print(f'Filtering high missing participants - removing', len(remove_participants), 'from dataset')
        filtered = data[~data.subject_id.isin(remove_participants)]       
        return filtered
    
    def add_missingness_indicator(self, data) -> pd.DataFrame:
        print('Adding missingness indicator variables')
        passive_feature_rep = {
            'location': 'f_loc:phone_locations_doryab_locationentropy:allday',
            'steps': 'f_steps:fitbit_steps_intraday_rapids_sumsteps:allday',
            'sleep': 'f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday',
            'call': 'f_call:phone_calls_rapids_outgoing_sumduration:allday',
        }
        n_feats = len(self.feature_cols)
        for var in self.sensor_data_types:
            if var not in passive_feature_rep.keys():
                print(f'\n ~~ WARNING {var} data type has no variable set to look for missingness ~~ \n')
                continue
            missing_indicator = f'{var}_missing'
            data[missing_indicator] = data[passive_feature_rep[var]].isna().astype(int)
            self.feature_cols.append(missing_indicator)

        print(f'\tAdded {len(self.feature_cols) - n_feats} missingness indicator variables, total features: {len(self.feature_cols)}')
        return data


    @staticmethod
    def get_phq_periods(data, features, period=1) -> pd.DataFrame:
        anomaly_detector_cols = [d for d in data.columns if d.endswith("_anomaly")]
        if len(anomaly_detector_cols) == 0:
            raise ValueError('No anomaly detector columns')
        
        phq = data[['subject_id', 'study_day', 'phq4']]\
                .dropna()\
                .sort_values(by=['subject_id', 'study_day'])\
                .drop_duplicates()\
                .reset_index(drop=True)

        results_dict = {
            'subject_id': [],
            'start': [],
            'stop': [],
            'days': [],
            'complete_days': [],
            'phq_start': [],
            'phq_stop': [],
            'phq_change': [],
            **{
                c: [] for c in anomaly_detector_cols
            },
        }

        for i, row in tqdm(phq.iterrows()):
            last_row = phq.iloc[i-period]
            if last_row['subject_id'] != row['subject_id']:
                continue

            anomalies = data.loc[
                (
                    (data.subject_id == row['subject_id']) &
                    (data.study_day > last_row['study_day']) &
                    (data.study_day <= row['study_day'])
                ),
                features + anomaly_detector_cols
            ]
            results_dict['subject_id'].append(row['subject_id'])
            results_dict['start'].append(last_row['study_day'])
            results_dict['stop'].append(row['study_day'])
            results_dict['days'] = row['study_day'] - last_row['study_day']
            results_dict['complete_days'].append(anomalies[features].dropna().shape[0])
            results_dict['phq_start'].append(last_row['phq4'])
            results_dict['phq_stop'].append(row['phq4'])
            results_dict['phq_change'].append(row['phq4'] - last_row['phq4'])
            for c in anomaly_detector_cols:
                if anomalies[c].isnull().all():
                    results_dict[c].append(np.nan)
                else:
                    results_dict[c].append(anomalies[c].sum())
                
        results = pd.DataFrame(results_dict)
        results['period'] = period
        return results


class CrossCheck(DatasetBase):
    def __init__(
        self,
        data_path: str = '~/Data/mHealth_external_datasets/CrossCheck/CrossCheck_Daily_Data.csv'
    ):
        if Path(data_path).expanduser().is_file():
            self.data_raw = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(
                f'{data_path} does not exist, set data_path to downloaded CrossCheck\
                     daily data from https://pbh.tech.cornell.edu/data.html'
            )
        

        # Separate features by type
        self.feature_cols = [
            f for f in self.data_raw.columns.values if f not in
            ['study_id', 'eureka_id', 'day', 'date', 'subject_id', 'study_day', 'first_day']
        ]

        # Ecological momentary assessment data
        self.ema_cols = [f for f in self.feature_cols if 'ema' in f]

        # Passive sensing data
        self.behavior_cols = [f for f in self.feature_cols if 'ema' not in f]

        # Preprocess data
        self.data = self.fill_empty_days(self.preprocess()).fillna(np.nan)

    @staticmethod
    def fill_empty_days(
        data: pd.DataFrame
    ) -> pd.DataFrame:
        expected_rows = []
        for (eid, sid), s_df in data.groupby(['eureka_id', 'subject_id']):
            min_d = s_df.study_day.min()
            max_d = s_df.study_day.max()
            n_days = 1 + max_d - min_d
            expected_rows.append(
                pd.DataFrame({
                    'subject_id': [sid] * n_days,
                    'eureka_id': [eid] * n_days,
                    'study_day': np.arange(min_d, max_d + 1)
                })
            )

            # Uncomment to show effect of adding rows
            # if expected_rows[-1].shape[0] != s_df.shape[0]:
            #     print('Empty rows added for:', sid, eid, expected_rows[-1].shape[0], s_df.shape[0])

        return data.merge(
            pd.concat(expected_rows),
            how='right',
            validate='1:1',
        )

    def preprocess(
        self,
    ) -> pd.DataFrame:
        # -1 id seems to be a testing account, has many ~25 eureka ids
        reform_data = self.data_raw[self.data_raw.study_id != -1].copy()
        reform_data['date'] = pd.to_datetime(
            reform_data['day'], format='%Y%m%d'
        )
        reform_data['subject_id'] = reform_data['study_id']

        # Exclude dates before the year 2000
        reform_data = reform_data[reform_data.day > 2*(10**7)]

        # Get first date of participant
        first_days = pd.DataFrame(
            reform_data.groupby('subject_id')['date'].min()
        ).reset_index().rename(columns={'date': 'first_day'})

        # Set study day as relative days in study
        reform_data = reform_data.merge(first_days, validate='m:1')
        reform_data['study_day'] = (
            reform_data['date'] - reform_data['first_day']
        ).dt.days

        # Clean data points
        sleep_act_cols = [
            c for c in self.behavior_cols
            if (c.startswith('sleep_') or c.startswith('act'))
        ]
        reform_data.loc[
            (
                reform_data['quality_activity'] < 2
            ),
            sleep_act_cols
        ] = np.nan
        reform_data.loc[
            (
                reform_data['quality_activity'].isnull()
            ),
            sleep_act_cols
        ] = np.nan

        return reform_data
