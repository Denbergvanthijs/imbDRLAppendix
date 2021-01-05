import pandas as pd
from tqdm import tqdm

hr = {220045}  # Heartrate
resp = {220210, 224422, 224689, 224690}  # Respiratory rate
temp = {223762, 223761}  # Temperature
scr = {50912, 51081, 220615}  # Serum Creatinine https://github.com/MIT-LCP/mimic-code/blob/master/notebooks/aline/aline_sofa.sql
gcsE = {220739}  # Glasscow Coma Scale Eyes
gcsM = {223900}  # Glasscow Coma Scale Movement
gcsV = {223901}  # Glasscow Coma Scale Verbal

dtypes = {"subject_id": int, "hadm_id": int, "itemid": int, "charttime": str, "valuenum": float}
df_total = pd.read_csv("./data/aki_allevents.csv", dtype=dtypes, parse_dates=["charttime"], infer_datetime_format=True)
print(f"{df_total.subject_id.nunique():_} total patients with relevant measurements; {df_total.hadm_id.nunique():_} total ICU admissions;")

# Only keep 5 <= Heartrate <= 350
print(f"{df_total.itemid.value_counts().loc[hr].sum():_} heartrate measurements;")
df_total = df_total.loc[~(df_total["itemid"].isin(hr) & ((df_total["valuenum"] < 5) | (df_total["valuenum"] > 350)))]
print(f"{df_total.itemid.value_counts().loc[hr].sum():_} valid heartrate measurements;")

print(f"{df_total.itemid.value_counts().loc[resp].sum():_} respiratory measurements;")
df_total = df_total.loc[~(df_total["itemid"].isin(resp) & ((df_total["valuenum"] < 1) | (df_total["valuenum"] > 150)))]
print(f"{df_total.itemid.value_counts().loc[resp].sum():_} valid respiratory measurements;")

print(f"{df_total.itemid.value_counts().loc[temp].sum():_} temperature measurements;")
df_total['valuenum'] = df_total.apply(lambda x: (x.valuenum - 32) / 1.8 if x.itemid == 223761 else x.valuenum, axis=1)  # F to C
df_total = df_total.loc[~(df_total["itemid"].isin(temp) & ((df_total["valuenum"] < 10) | (df_total["valuenum"] > 50)))]
print(f"{df_total.itemid.value_counts().loc[temp].sum():_} valid temperature measurements;")

print(f"{df_total.itemid.value_counts().loc[scr].sum():_} Serum Creatinine measurements;")
df_total = df_total.loc[~(df_total["itemid"].isin(scr) & ((df_total["valuenum"] < 0) | (df_total["valuenum"] > 50)))]
print(f"{df_total.itemid.value_counts().loc[scr].sum():_} valid Serum Creatinine measurements;")

print(f"{df_total.itemid.value_counts().loc[gcsE | gcsM | gcsV].sum():_} Glasscow Coma Scale measurements;")
df_total = df_total.loc[~(df_total["itemid"].isin(gcsE) & ((df_total["valuenum"] < 1) | (df_total["valuenum"] > 4)))]
df_total = df_total.loc[~(df_total["itemid"].isin(gcsM) & ((df_total["valuenum"] < 1) | (df_total["valuenum"] > 6)))]
df_total = df_total.loc[~(df_total["itemid"].isin(gcsV) & ((df_total["valuenum"] < 1) | (df_total["valuenum"] > 5)))]
print(f"{df_total.itemid.value_counts().loc[gcsE | gcsM | gcsV].sum():_} valid Glasscow Coma Scale measurements;")

for c, i in enumerate(tqdm((hr, resp, temp, scr, gcsE, gcsM, gcsV))):  # Set each category to a single integer
    df_total["itemid"] = df_total["itemid"].replace(list(i), c)

# Only keep the hospital admissions with at least one measurement for each proxy: 7 proxies => 7 unique itemids
df_total = df_total[df_total.groupby(["subject_id", "hadm_id"])["itemid"].transform("nunique") == 7]
df_total.to_csv("./data/aki_alleventscleaned.csv", index=False)
print(f"{df_total.subject_id.nunique():_} total patients with at least 1 measurement for each proxy; "
      f"{df_total.hadm_id.nunique():_} total ICU admissions;")
