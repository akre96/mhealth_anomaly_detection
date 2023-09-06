import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Literal
from tqdm.auto import tqdm
import json


class DatasetBase:
    def __init__(
        self,
        data_path: str,
    ):
        if Path(data_path).expanduser().is_file():
            self.data_raw = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(f"{data_path} does not exist")

        self.data = self.preprocess(self.data_raw)
        self.id_cols = []

    def addStudyDay(
        self, data, first_days: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        # Get first date of participant
        if first_days is None:
            first_days = (
                pd.DataFrame(data.groupby("subject_id")["date"].min())
                .reset_index()
                .rename(columns={"date": "first_day"})
            )

        # Set study day as relative days in study
        reform_data = data.merge(first_days, validate="m:1")
        reform_data["study_day"] = (
            reform_data["date"] - reform_data["first_day"]
        ).dt.days
        self.id_cols = self.id_cols + ["study_day"]
        return reform_data

    def preprocess(self, data_raw) -> pd.DataFrame:
        return data_raw


class BRIGHTEN_v2(DatasetBase):
    # Data downloaded from https://www.synapse.org/#!Synapse:syn10848316
    # tables saved as csv with no metadata
    def __init__(
        self,
        data_path: str | Path,
        feature_types: List = ["mobility", "phone_communication", "weather"],
    ):
        if not Path(data_path).expanduser().is_dir():
            raise FileNotFoundError(
                f"{data_path} must be a folder to BIRGHTEN data"
            )
        self.data_path = Path(data_path)
        self.feature_types = feature_types
        self.features = []
        self.id_cols = ["subject_id", "date", "week"]
        self.data_raw = self.getPassiveSensorData()
        self.data = self.preprocess(self.data_raw)

    def preprocess(self, data_raw) -> pd.DataFrame:
        data = self.filterByStudyLength(data_raw, min_weeks=8)
        data = self.addStudyDay(data)
        data = fillEmptyDays(data)
        data["came_to_work"] = data["came_to_work"].astype(bool).astype(float)
        return data

    def getTrainTestSplit(self):
        train_test_split = pd.read_csv(
            Path(self.data_path, "BRIGHTEN_v2_train_test_split.txt"), sep="\t"
        )
        # Batch column contains train test splitting
        # Batch 1 = train, Batch 2 = test
        train_test_split["batch"] = train_test_split.batch.astype("category")
        train_test_split["marital_status_short"] = train_test_split[
            "marital_status"
        ].map(
            {
                "Married/Partner": "Married/Partner",
                "Separated/Widowed/Divorced": "Sep/Wid/Div",
                "Single": "Single",
            }
        )
        return train_test_split

    # Gets command for running ARTS.pl package to generate Train/Test Split
    def generateARTSParams(self):
        split_df = self.getParticipantSplitParameters()
        param_cols = [str(i) for i in range(2, len(split_df.columns))]
        ARTS_param_c = ",".join(param_cols) + ";" + ";".join(param_cols)
        train_n = round(split_df.shape[0] * 0.7)
        test_n = split_df.shape[0] - train_n

        # Where numeric columns
        is_numeric = np.where(split_df.dtypes.isin([int, np.float64]))[0]
        ARTS_param_cc = ",".join([str(i + 1) for i in is_numeric])
        ARTS_param_b = f"{train_n},{test_n}"

        input_path = Path(self.data_path, "BRIGHTEN_v2_stratify.txt")
        output_path = Path(self.data_path, "BRIGHTEN_v2_train_test_split.txt")
        print("Saving input to ARTS to", input_path)
        split_df.to_csv(input_path, sep="\t", index=False)

        return f'src/ARTS.pl \\\n\t-i f{input_path} \\\n\t-o f{output_path} \\\n\t-c "{ARTS_param_c}" \\\n\t-b {ARTS_param_b} \\\n\t-cc {ARTS_param_cc}'

    # Get demographic data
    def getDemographics(self) -> pd.DataFrame:
        demog = pd.read_csv(
            Path(self.data_path, "demographics.csv"), parse_dates=["startdate"]
        )
        demog = demog.rename(columns={"participant_id": "subject_id"})
        cat_cols = [
            "gender",
            "education",
            "working",
            "income_satisfaction",
            "income_lastyear",
            "marital_status",
            "race",
            "heard_about_us",
            "device",
            "study_arm",
            "study",
        ]
        demog[cat_cols] = demog[cat_cols].astype("category")
        demo_v2 = demog[demog.study == "Brighten-v2"]
        return demo_v2

    # Get self report data
    def getSelfReport(
        self, self_report: Literal["GAD7", "sleep_quality", "SDS", "phq9"]
    ) -> pd.DataFrame:
        sr_cols = None
        if self_report == "phq9":
            return self.getPHQ9()
        elif self_report == "SDS":
            sr_cols = ["sds_1", "sds_2", "sds_3", "stress", "support"]
        elif self_report == "GAD7":
            sr_cols = [f"gad7_{i}" for i in range(1, 9)] + ["gad7_sum"]
        elif self_report == "sleep_quality":
            sr_cols = [f"sleep_{i}" for i in range(1, 4)]
        else:
            raise ValueError(f"Unknown self report {self_report}")
        sr = pd.read_csv(Path(self.data_path, f"{self_report}.csv"))
        sr["date"] = pd.to_datetime(pd.to_datetime(sr["dt_response"]).dt.date)
        sr = sr.drop(columns=["dt_response"]).rename(
            columns={"participant_id": "subject_id"}
        )
        sr_dedup = sr[["subject_id", "date"] + sr_cols].drop_duplicates(
            subset=["subject_id", "date"], keep="last"
        )
        sr_dedup["self_report"] = self_report
        print(
            f"{self_report} duplicates:",
            sr.shape[0] - sr_dedup.shape[0],
            "of",
            sr.shape[0],
        )
        return sr_dedup

    def getPHQ9(self) -> pd.DataFrame:
        baseline_phq9 = pd.read_csv(Path(self.data_path, "baseline_phq9.csv"))
        baseline_phq9 = baseline_phq9.rename(
            columns={
                "participant_id": "subject_id",
                "baselinePHQ9date": "date",
            }
        )
        phq_items = [f"phq9_{i}" for i in range(1, 10)]
        baseline_phq9["sum_phq9"] = baseline_phq9[phq_items].sum(axis=1)
        baseline_phq9_v2 = baseline_phq9[
            baseline_phq9.study == "Brighten-v2"
        ].drop(columns=["study"])
        baseline_phq9_v2["baseline"] = True

        phq9 = pd.read_csv(Path(self.data_path, "phq9.csv"))
        phq9 = phq9.rename(
            columns={"participant_id": "subject_id", "phq9Date": "date"}
        )
        phq9["baseline"] = False
        phq9_v2 = phq9[
            phq9.subject_id.isin(baseline_phq9_v2.subject_id.unique())
        ]

        # Merge baseline and weekly phq9
        merged_phq9 = pd.concat([phq9_v2, baseline_phq9_v2])
        if (
            merged_phq9.shape[0]
            != phq9_v2.shape[0] + baseline_phq9_v2.shape[0]
        ):
            print(
                merged_phq9.shape[0],
                phq9_v2.shape[0],
                baseline_phq9_v2.shape[0],
            )
            raise ValueError("Shape mismatch")
        merged_phq9["date"] = pd.to_datetime(merged_phq9["date"])

        phq9_dedup = merged_phq9.drop_duplicates(
            subset=["subject_id", "date"], keep="last"
        )
        print(
            "PHQ9 duplicates:",
            merged_phq9.shape[0] - phq9_dedup.shape[0],
            "of",
            merged_phq9.shape[0],
        )
        phq9_dedup["self_report"] = "phq9"
        return phq9_dedup

    # Get data used to generate a train-test split of all data
    def getParticipantSplitParameters(self) -> pd.DataFrame:
        phq9 = self.getPHQ9()
        baseline_phq9 = phq9[phq9.baseline]
        demographics = self.getDemographics()
        data_subjects = pd.DataFrame(
            self.data[["subject_id"]]
        ).drop_duplicates()

        stratify_data = data_subjects.merge(
            baseline_phq9[["subject_id", "sum_phq9"]],
            how="inner",
        )
        print(
            "missing PHQ from:",
            data_subjects.subject_id.nunique()
            - stratify_data.subject_id.nunique(),
        )

        strat_cols_demog = [
            "gender",
            "age",
            "education",
            "working",
            "income_satisfaction",
            "income_lastyear",
            "marital_status",
            "race",
            "device",
            "study_arm",
        ]
        stratify_data = stratify_data.merge(
            demographics[["subject_id"] + strat_cols_demog],
            how="inner",
        )
        print(
            "missing demographics from:",
            data_subjects.subject_id.nunique()
            - stratify_data.subject_id.nunique(),
        )

        return stratify_data

    def getPassiveSensorData(self):
        id_cols = ["participant_id", "dt_passive", "week"]
        features_df = pd.DataFrame(columns=id_cols)
        for p in self.feature_types:
            df = pd.read_csv(Path(self.data_path, p + "_v2.csv"))
            features_df = features_df.merge(
                df,
                how="outer",
            )
        self.features = [c for c in features_df.columns if c not in id_cols]
        features_df["dt_passive"] = pd.to_datetime(features_df["dt_passive"])
        return features_df.rename(
            columns={"participant_id": "subject_id", "dt_passive": "date"}
        )

    def filterByStudyLength(self, data: pd.DataFrame, min_weeks: int = 8):
        weeks = data.groupby("subject_id").week.max()
        keep_ids = weeks[weeks >= min_weeks].index
        return data[data.subject_id.isin(keep_ids)].copy()


class OPTIMA(DatasetBase):
    def addDay1Date(self, data: pd.DataFrame) -> pd.DataFrame:
        day1 = (
            data.groupby("subject_id")
            .date.min()
            .reset_index()
            .rename(columns={"date": "day1"})
        )
        data_fmt = data.merge(day1, how="left")
        data_fmt["study_day"] = (data_fmt.date - data_fmt.day1).dt.days
        return data_fmt

    def getSurveyData(
        self,
        redcap_data_path: Path,
        eid_uid_path: Path,
        survey_info_path: Path = Path("lib/surveys.json"),
    ) -> pd.DataFrame:
        with open(survey_info_path, "r") as f:
            surveys = json.load(f)
        eid_uid = pd.read_csv(eid_uid_path)
        eid_uid.columns = ["record_id", "eid"]
        redcap_data = (
            pd.concat([pd.read_csv(f) for f in redcap_data_path.glob("*.csv")])
            .merge(eid_uid, how="left")
            .rename(columns={"eid": "user_id"})
        )

        taken_cols = []
        survey_res = []
        for _, params in surveys.items():
            survey = params["name"]
            start = params["start"]
            duration = params["duration"]
            prefix = params["prefix"]
            if "suffix" in params.keys():
                suffix = params["suffix"]
            else:
                suffix = None
            if suffix is not None:
                survey_cols = [
                    c
                    for c in redcap_data.columns
                    if c.startswith(prefix)
                    and c.endswith(suffix)
                    and not c.endswith("_end")
                ]
            else:
                survey_cols = [
                    c
                    for c in redcap_data.columns
                    if c.startswith(prefix)
                    and not c in taken_cols
                    and not c.endswith("_end")
                ]
            if start in survey_cols:
                survey_cols.remove(start)

            survey_data = redcap_data[
                survey_cols + [start, "redcap_event_name", "user_id"]
            ].dropna(subset=[start])
            survey_data = survey_data.rename(
                columns={
                    start: "survey_start",
                }
            )
            for c in survey_cols + [start]:
                taken_cols.append(c)

            survey_data["survey"] = survey
            survey_data["duration"] = duration
            survey_data = survey_data.melt(
                id_vars=[
                    "user_id",
                    "survey",
                    "survey_start",
                    "duration",
                    "redcap_event_name",
                ],
                value_name="response",
                var_name="question",
            )
            if suffix is not None:
                survey_data["question"] = survey_data["question"].str.replace(
                    suffix, ""
                )
            survey_res.append(survey_data)
        survey_df = pd.concat(survey_res).reset_index(drop=True)
        survey_df["survey_start"] = pd.to_datetime(survey_df["survey_start"])
        survey_df["response"] = pd.to_numeric(
            survey_df["response"], errors="coerce"
        )
        return survey_df

    def preprocess(self, data_raw) -> pd.DataFrame:
        data_raw = data_raw.rename(columns={"user_id": "subject_id"})
        data_raw["date"] = pd.to_datetime(data_raw["date"])
        self.sensor_cols = [
            c
            for c in data_raw.columns
            if c not in ["subject_id", "date", "study_day"]
        ]
        hk_data_fmt = self.addDay1Date(data_raw)
        hk_data_fmt = hk_data_fmt[
            ["subject_id", "study_day", "date", *self.sensor_cols]
        ]
        hk_data_fmt = fillEmptyDays(hk_data_fmt)
        return hk_data_fmt


class GLOBEM(DatasetBase):
    def __init__(
        self,
        data_path: str = "~/Data/mHealth_external_datasets/GLOBEM",
        year: int = 2,
        sensor_data_types: List = [
            "wifi",
            "steps",
            "sleep",
            "screen",
            "rapids",
            "location",
            "call",
            "bluetooth",
        ],
        load_data: bool = True,
    ):
        if year not in [2, 3, 4]:
            raise ValueError(
                "Year for the GLOBEM dataset analysis must be 2, 3, or 4"
            )

        self.possible_sensor_types = [
            "wifi",
            "steps",
            "sleep",
            "screen",
            "rapids",
            "location",
            "call",
            "bluetooth",
        ]
        for t in sensor_data_types:
            if t not in self.possible_sensor_types:
                raise ValueError(f"{t} not a possible sensor type in GLOBEM")

        if not Path(data_path).expanduser().is_dir():
            raise FileNotFoundError(
                f"{data_path} does not exist, set data_path to unzipped\
                     GLOBEM dataset from\
                     https://physionet.org/content/globem/1.0"
            )
        else:
            self.data_path = data_path

        self.sensor_data_types = sensor_data_types
        self.year = year
        self.id_cols = ["pid", "platform", "date"]
        self.feature_cols = []

        if load_data:
            self.data_raw = self.combine_data()
            self.data = self.preprocess(self.data_raw)

            self.sensor_cols = [f for f in self.feature_cols if f != "phq4"]

    def preprocess(self, data_raw) -> pd.DataFrame:
        data = self.filter_high_missing_cols(data_raw).rename(
            columns={
                "pid": "subject_id",
            }
        )
        self.id_cols = ["subject_id", "platform", "date"]
        data = self.filter_redundant_cols(data)
        data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
        data = self.addStudyDay(data)
        data = fillEmptyDays(data)
        data = self.filter_high_missing_participants(data)
        data = self.add_missingness_indicator(data)

        return data[self.id_cols + self.feature_cols]

    def combine_data(self) -> pd.DataFrame:
        surveys = self.get_weekly_phq4()
        sensors = self.get_daily_sensor_data()
        participants = self.get_participant_info()

        participant_sensors = participants.merge(
            sensors, how="inner", validate="1:m"
        )

        data = participant_sensors.merge(
            surveys, how="outer", validate="1:1"
        ).sort_values(by=["pid", "date"])

        return data

    def get_pre_post_surveys(self) -> pd.DataFrame:
        pre_path = Path(
            self.data_path, f"INS-W_{self.year}", "SurveyData", "pre.csv"
        )

        pre = pd.read_csv(pre_path, index_col=0).rename(
            columns={"date": "pre_date"}
        )
        pre_col_names = [c.removesuffix("_PRE") for c in pre.columns]

        post_path = Path(
            self.data_path, f"INS-W_{self.year}", "SurveyData", "post.csv"
        )

        post = pd.read_csv(post_path, index_col=0).rename(
            columns={"date": "post_date"}
        )
        post_col_names = [c.removesuffix("_POST") for c in post.columns]

        prepost = pre.merge(post, how="inner", on="pid").rename(
            columns={"pid": "subject_id"}
        )
        for col in pre_col_names:
            if col in post_col_names and col not in [
                "pid",
                "pre_date",
                "post_date",
            ]:
                prepost[f"{col}_CHANGE"] = (
                    prepost[f"{col}_POST"] - prepost[f"{col}_PRE"]
                )

        return prepost

    def get_weekly_phq4(self) -> pd.DataFrame:
        weekly_survey_path = Path(
            self.data_path,
            f"INS-W_{self.year}",
            "SurveyData",
            "dep_weekly.csv",
        )

        return pd.read_csv(weekly_survey_path, index_col=0)[
            ["pid", "date", "phq4"]
        ].dropna()

    def get_daily_sensor_data(self) -> pd.DataFrame:
        data = pd.DataFrame()
        for i, sensor_type in enumerate(self.sensor_data_types):
            sensor_data_path = Path(
                self.data_path,
                f"INS-W_{self.year}",
                "FeatureData",
                f"{sensor_type}.csv",
            )
            df = pd.read_csv(sensor_data_path, index_col=0)
            if i == 0:
                data = df
            else:
                data = data.merge(df, how="outer", validate="1:1")

        return self.filter_nondaily_cols(data)

    def get_participant_info(self) -> pd.DataFrame:
        info_path = Path(
            self.data_path,
            f"INS-W_{self.year}",
            "ParticipantsInfoData",
            "platform.csv",
        )
        return pd.read_csv(info_path, index_col=0).reset_index()

    def filter_nondaily_cols(self, data) -> pd.DataFrame:
        cols = [c for c in data.columns if c not in self.id_cols]
        use_segments = [
            "morning",
            "afternoon",
            "evening",
            "night",
            "allday",
        ]
        filtered = [c for c in cols if c.split(":")[-1] in use_segments]
        print(
            f"Filtering only daily features. Going from {len(cols)} to {len(filtered)} features"
        )
        return data[["pid", "date"] + filtered]

    def filter_high_missing_cols(self, data) -> pd.DataFrame:
        missing = data.isnull().sum().sort_values().T.reset_index()
        # Filter those more missing than PHQ 4
        thresh = missing[missing["index"] == "phq4"][0].values[0]
        use_cols = list(missing[missing[0] <= thresh]["index"].values)

        print(
            f"Filtering columns with high missingness. Going from {missing.shape[0]} to {len(use_cols)} features"
        )
        self.feature_cols = [c for c in use_cols if c not in self.id_cols]
        return data[use_cols]

    def filter_redundant_cols(self, data) -> pd.DataFrame:
        feats = self.feature_cols
        # Remove wake time since sleep time present
        filtered = [c for c in feats if "awake" not in c]

        # Only look at raw data, not normalized or discretized
        redundant_steps = [
            "maxsumsteps",
            "minsumsteps",
            "avgsumsteps",
            "mediansumsteps",
            "stdsumsteps",
            "maxsteps",
            "minsteps",
            "avgsteps",
            "stdsteps",
        ]
        filtered = [
            c
            for c in filtered
            if not (
                len(c) > 5
                and (
                    c.split(":")[-2].endswith("_norm")
                    or c.split(":")[-2].endswith("_dis")
                    or c.split(":")[-2].split("_")[-1] in redundant_steps
                )
            )
        ]
        print(
            f"Filtering redundant features. Going from {len(feats)} to {len(filtered)} features"
        )
        self.feature_cols = filtered
        return data[self.id_cols + filtered]

    def filter_high_missing_participants(self, data) -> pd.DataFrame:
        remove_participants = []
        if self.year == 2:
            remove_participants = [
                "INS-W_314",
                "INS-W_316",
                "INS-W_317",
                "INS-W_322",
                "INS-W_335",
                "INS-W_361",
                "INS-W_392",
                "INS-W_394",
                "INS-W_421",
                "INS-W_436",
                "INS-W_460",
                "INS-W_479",
                "INS-W_485",
                "INS-W_493",
                "INS-W_495",
                "INS-W_497",
                "INS-W_505",
                "INS-W_512",
                "INS-W_523",
                "INS-W_527",
                "INS-W_536",
                "INS-W_537",
                "INS-W_555",
                "INS-W_556",
                "INS-W_569",
                "INS-W_570",
            ]
        elif self.year == 3:
            remove_participants = [
                "INS-W_601",
                "INS-W_606",
                "INS-W_637",
                "INS-W_642",
                "INS-W_653",
                "INS-W_679",
                "INS-W_681",
                "INS-W_737",
                "INS-W_756",
            ]
        else:
            print(
                "Warning: Filtering participants not established for years 1 and 4"
            )

        print(
            "Filtering high missing participants - removing",
            len(remove_participants),
            "from dataset",
        )
        filtered = data[~data.subject_id.isin(remove_participants)]
        return filtered

    def add_missingness_indicator(self, data) -> pd.DataFrame:
        print("Adding missingness indicator variables")
        passive_feature_rep = {
            "location": "f_loc:phone_locations_doryab_locationentropy:allday",
            "steps": "f_steps:fitbit_steps_intraday_rapids_sumsteps:allday",
            "sleep": "f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday",
            "call": "f_call:phone_calls_rapids_outgoing_sumduration:allday",
        }
        n_feats = len(self.feature_cols)
        for var in self.sensor_data_types:
            if var not in passive_feature_rep.keys():
                print(
                    f"\n ~~ WARNING {var} data type has no variable set to look for missingness ~~ \n"
                )
                continue
            missing_indicator = f"{var}_missing"
            data[missing_indicator] = (
                data[passive_feature_rep[var]].isna().astype(int)
            )
            self.feature_cols.append(missing_indicator)

        print(
            f"\tAdded {len(self.feature_cols) - n_feats} missingness indicator variables, total features: {len(self.feature_cols)}"
        )
        return data

    @staticmethod
    def get_phq_periods(data, features, period=1) -> pd.DataFrame:
        anomaly_detector_cols = [
            d for d in data.columns if d.endswith("_anomaly")
        ]
        if len(anomaly_detector_cols) == 0:
            raise ValueError("No anomaly detector columns")

        phq = (
            data[["subject_id", "study_day", "phq4"]]
            .dropna()
            .sort_values(by=["subject_id", "study_day"])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        results_dict = {
            "subject_id": [],
            "start": [],
            "stop": [],
            "days": [],
            "complete_days": [],
            "phq_start": [],
            "phq_stop": [],
            "phq_change": [],
            **{c: [] for c in anomaly_detector_cols},
        }

        for i, row in tqdm(phq.iterrows()):
            last_row = phq.iloc[i - period]
            if last_row["subject_id"] != row["subject_id"]:
                continue
            if last_row["study_day"] > row["study_day"]:
                continue

            anomalies = data.loc[
                (
                    (data.subject_id == row["subject_id"])
                    & (data.study_day > last_row["study_day"])
                    & (data.study_day <= row["study_day"])
                ),
                features + anomaly_detector_cols,
            ]
            results_dict["subject_id"].append(row["subject_id"])
            results_dict["start"].append(last_row["study_day"])
            results_dict["stop"].append(row["study_day"])
            results_dict["days"] = row["study_day"] - last_row["study_day"]
            results_dict["complete_days"].append(
                anomalies[features].dropna().shape[0]
            )
            results_dict["phq_start"].append(last_row["phq4"])
            results_dict["phq_stop"].append(row["phq4"])
            results_dict["phq_change"].append(row["phq4"] - last_row["phq4"])
            for c in anomaly_detector_cols:
                if anomalies[c].isnull().all():
                    results_dict[c].append(np.nan)
                else:
                    results_dict[c].append(anomalies[c].sum())

        results = pd.DataFrame(results_dict)
        results["period"] = period
        return results


class CrossCheck(DatasetBase):
    def __init__(
        self,
        data_path: str = "~/Data/mHealth_external_datasets/CrossCheck/CrossCheck_Daily_Data.csv",
    ):
        if Path(data_path).expanduser().is_file():
            self.data_raw = pd.read_csv(data_path)
        else:
            raise FileNotFoundError(
                f"{data_path} does not exist, set data_path to downloaded CrossCheck\
                     daily data from https://pbh.tech.cornell.edu/data.html"
            )

        # Separate features by type
        self.feature_cols = [
            f
            for f in self.data_raw.columns.values
            if f
            not in [
                "study_id",
                "eureka_id",
                "day",
                "date",
                "subject_id",
                "study_day",
                "first_day",
            ]
        ]

        # Ecological momentary assessment data
        self.ema_cols = [f for f in self.feature_cols if "ema" in f]

        # Passive sensing data
        self.behavior_cols = [f for f in self.feature_cols if "ema" not in f]

        # Preprocess data
        self.data = fillEmptyDays(self.preprocess()).fillna(np.nan)

    def preprocess(
        self,
    ) -> pd.DataFrame:
        # -1 id seems to be a testing account, has many ~25 eureka ids
        reform_data = self.data_raw[self.data_raw.study_id != -1].copy()
        reform_data["date"] = pd.to_datetime(
            reform_data["day"], format="%Y%m%d"
        )
        reform_data["subject_id"] = reform_data["study_id"]

        # Exclude dates before the year 2000
        reform_data = reform_data[reform_data.day > 2 * (10**7)]

        # Get first date of participant
        first_days = (
            pd.DataFrame(reform_data.groupby("subject_id")["date"].min())
            .reset_index()
            .rename(columns={"date": "first_day"})
        )

        # Set study day as relative days in study
        reform_data = reform_data.merge(first_days, validate="m:1")
        reform_data["study_day"] = (
            reform_data["date"] - reform_data["first_day"]
        ).dt.days

        # Clean data points
        sleep_act_cols = [
            c
            for c in self.behavior_cols
            if (c.startswith("sleep_") or c.startswith("act"))
        ]
        reform_data.loc[
            (reform_data["quality_activity"] < 2), sleep_act_cols
        ] = np.nan
        reform_data.loc[
            (reform_data["quality_activity"].isnull()), sleep_act_cols
        ] = np.nan

        return reform_data


def fillEmptyDays(data: pd.DataFrame) -> pd.DataFrame:
    expected_rows = []
    for (sid), s_df in data.groupby("subject_id"):
        min_d = s_df.study_day.min()
        max_d = s_df.study_day.max()
        n_days = 1 + max_d - min_d
        expected_rows.append(
            pd.DataFrame(
                {
                    "subject_id": [sid] * n_days,
                    "study_day": np.arange(min_d, max_d + 1),
                }
            )
        )

    filled = data.merge(
        pd.concat(expected_rows),
        how="right",
        validate="1:1",
    )

    print(
        "Filling empty study days with NaN values going from",
        data.shape[0],
        "to",
        filled.shape[0],
        ". Added ",
        filled.shape[0] - data.shape[0],
    )
    return filled
