import pandas as pd
from tqdm import tqdm

""""
This script collects all relevant lab and chart events and combines them.
Only events from relevant patients and with relevant itemids are kept.
The file `aki_allevents.csv` contains all relevant itemids from both lab- and chartevents.
Columns: subject_id, hadm_id, itemid, charttime, valuenum

Input:  ./data/aki_1.csv (from `1_selectPatients.py`)
        ./mimic/icu/chartevents.csv
        ./mimic/hosp/labevents.csv
Output: ./data/aki_chartevents.csv
        ./data/aki_labevents.csv
        ./data/aki_allevents.csv
"""

hr = {220045}  # Heartrate
resp = {220210, 224422, 224689, 224690}  # Respiratory rate
temp = {223762, 223761}  # Temperature
scr = {50912, 51081, 220615}  # Serum Creatinine https://github.com/MIT-LCP/mimic-code/blob/master/notebooks/aline/aline_sofa.sql
gcsE = {220739}  # Glasscow Coma Scale Eyes
gcsM = {223900}  # Glasscow Coma Scale Movement
gcsV = {223901}  # Glasscow Coma Scale Verbal
all_itemids = (*hr, *resp, *temp, *scr, *gcsE, *gcsM, *gcsV)

usecols = ("subject_id", "hadm_id", "itemid", "charttime", "valuenum")
types_patients = {"subject_id": int, "anchor_age": int, "hadm_id": int, "los": float}
types_cha = {"subject_id": int, "hadm_id": int, "itemid": int, "charttime": str, "valuenum": float}
types_lab = {"subject_id": int, "hadm_id": str, "itemid": int, "charttime": str, "valuenum": float}
parse = ["charttime"]

df_patients = pd.read_csv("./data/aki_1.csv", dtype=types_patients)
all_patients = df_patients.hadm_id.unique()
print(f"{df_patients.subject_id.nunique():_} patients; {df_patients.hadm_id.nunique():_} total ICU admissions;")

all_chunks = []
for chunk in tqdm(pd.read_csv("./mimic-iv-0.4/icu/chartevents.csv", usecols=usecols, chunksize=1_000_000, dtype=types_cha,
                              parse_dates=parse, infer_datetime_format=True), total=328):
    chunk = chunk.loc[chunk["hadm_id"].isin(all_patients)]  # Only include events on relevant patients
    chunk = chunk.loc[chunk["itemid"].isin(all_itemids)]  # Only include selected itemids
    all_chunks.append(chunk)

df_chart = pd.concat(all_chunks)  # Combine all chartevent chunks
df_chart.to_csv("./data/aki_chartevents.csv", index=False)
print(f"{df_chart.subject_id.nunique():_} chartevent patients; "
      f"{df_chart.hadm_id.nunique():_} chartevent ICU admissions; {df_chart.shape[0]:_} total chartevents;")

all_chunks = []
for chunk in tqdm(pd.read_csv("./mimic-iv-0.4/hosp/labevents.csv", usecols=usecols, chunksize=1_000_000, dtype=types_lab,
                              parse_dates=parse, infer_datetime_format=True), total=123):
    chunk.dropna(inplace=True)  # Drop all lab events without hospital admission
    chunk["hadm_id"] = chunk["hadm_id"].astype(int)
    chunk = chunk.loc[chunk["hadm_id"].isin(all_patients)]  # Only include events on relevant patients
    chunk = chunk.loc[chunk["itemid"].isin(all_itemids)]  # Only include selected itemids
    all_chunks.append(chunk)

df_labs = pd.concat(all_chunks)  # Combine all labevent chunks
df_labs.to_csv("./data/aki_labevents.csv", index=False)
print(f"{df_labs.subject_id.nunique():_} labevents patients; "
      f"{df_labs.hadm_id.nunique():_} labevents ICU admissions; {df_labs.shape[0]:_} total labevents;")

df_total = pd.concat((df_labs, df_chart))
df_total.to_csv("./data/aki_allevents.csv", index=False)
print(f"{df_total.subject_id.nunique():_} total patients; "
      f"{df_total.hadm_id.nunique():_} total ICU admissions; {df_total.shape[0]} total events;")
