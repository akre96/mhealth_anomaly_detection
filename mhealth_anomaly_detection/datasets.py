import pandas as pd
import numpy as np
from pathlib import Path



class CrossCheck:
    def __init__(
        self,
        data_path: str = '~/Data/CrossCheck/CrossCheck_Daily_Data.csv'
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
            if expected_rows[-1].shape[0] != s_df.shape[0]:
                print('Empty rows added for:', sid, eid, expected_rows[-1].shape[0], s_df.shape[0])

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
