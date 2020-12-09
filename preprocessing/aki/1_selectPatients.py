import pandas as pd

"""
This script generates the file "aki_1.csv"
The output .csv-file contains all patients including hadm_id, los, age, who stayed 5 to 1_000 hours on the ICU

Input:  ./mimic/core/patients.csv;
        ./mimic/icu/icustays.csv
Output: ./data/aki_1.csv
"""

# All patients: patient ID and patient approx. age
df_patients = pd.read_csv("./mimic-iv-0.4/core/patients.csv", dtype=int, usecols=["subject_id", "anchor_age"])
print(f"{df_patients.subject_id.nunique():_} patients;")

df_patients = df_patients.loc[df_patients["anchor_age"] >= 18]  # All patients >= 18
print(f"{df_patients.subject_id.nunique():_} patients >= 18;")

types = {"subject_id": int, "hadm_id": int, "los": float}
cols = ["subject_id", "hadm_id", "los"]  # los: length of stay in days
df_icu = pd.read_csv("./mimic-iv-0.4/icu/icustays.csv", dtype=types, usecols=cols)
print(f"{df_icu.subject_id.nunique():_} patients on ICU; {df_icu.hadm_id.nunique():_} total ICU admissions;")

# Filter all patients with a length of stay of 5 to 1_000 hours
df_icu = df_icu.loc[(df_icu.los >= 0.21) & (df_icu.los <= 41.67)]
print(f"{df_icu.subject_id.nunique():_} patients on ICU for 5 to 1_000 hours; {df_icu.hadm_id.nunique():_} total ICU admissions;")

df_total = df_patients.merge(df_icu, on=['subject_id'], how="inner")
print(f"{df_total.subject_id.nunique():_} patients; {df_total.hadm_id.nunique():_} total ICU admissions;")

df_total.to_csv("./data/aki_1.csv", index=False)
print(df_total.head(3))
