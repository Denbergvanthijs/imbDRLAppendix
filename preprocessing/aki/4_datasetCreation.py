import numpy as np
import pandas as pd
from imbDRL.utils import imbalance_ratio

df_total = pd.read_csv("./data/aki_alleventscleaned.csv", dtype={"subject_id": int, "hadm_id": int, "itemid": int, "charttime": str, "valuenum": float},
                       parse_dates=["charttime"], infer_datetime_format=True)
print(f"{df_total.subject_id.nunique():_} total patients with relevant parameters; {df_total.hadm_id.nunique():_} total ICU admissions;")

# Time after the first measurement per hospitalisation
df_total["timeafterepoch"] = df_total["charttime"] - df_total.groupby(["subject_id", "hadm_id"]).charttime.transform("min")
print(f"Latest recording of measurement after {df_total.timeafterepoch.max()} of hospital admission;")

# Number of hours after the first measurement per hospitalisation
# Number of days * 24 plus number of hours
interpolate_every = 4
df_total["hoursafterepoch"] = (df_total.timeafterepoch.dt.days * 24 + df_total.timeafterepoch.dt.seconds / 60 // 60) // interpolate_every

# Drop excess columns
df_total.drop(columns=["subject_id", "charttime", "timeafterepoch"], inplace=True)
df_total["hoursafterepoch"] = df_total["hoursafterepoch"].astype(int)
print(f"Latest recording of measurement after {df_total.hoursafterepoch.max():_} hours of hospital admission;")

# Limit measurements to first 72 hours
print(f"{df_total.shape[0]:_} measurements;")
df_total = df_total[df_total.hoursafterepoch <= (72 // interpolate_every - 1)]
print(f"{df_total.shape[0]:_} measurements within the first 72 hours of hospital admission;")

dfs = []
for c, name in enumerate(("hr", "resp", "temp", "scr", "gcsE", "gcsM", "gcsV")):
    # Dataframe with all measurements for only the current itemid
    df_temp = df_total.loc[df_total.itemid == c].drop(columns="itemid").groupby(["hadm_id", "hoursafterepoch"]).mean().unstack()
    # Resetting multi-index columns to interval 1 <= x < 72
    df_temp.columns = np.arange((72 // interpolate_every))
    df_temp = df_temp.add_prefix(name)  # E.g.: hr0, hr1, hr2, etc...

    print(f"{(total := df_temp.isna().sum().sum()):_} ({total / df_temp.size:.4f} %) missing values for {name};", end="\t")
    df_temp = df_temp.interpolate(axis=1, limit_direction="both")
    print(f"{df_temp.isna().sum().sum():_} missing values for {name} after interpolation;")

    dfs.append(df_temp)

df_complete = pd.concat(dfs, axis=1)

df_diag = pd.read_csv("./mimic-iv-0.4/hosp/diagnoses_icd.csv", usecols=["hadm_id", "icd_code"])
df_diag = df_diag.loc[df_diag.icd_code.str.strip().isin(["5849", "N179"])]

df_complete["aki"] = 0  # Default is no AKI
df_complete.loc[df_complete.index.isin(df_diag.hadm_id.unique()), "aki"] = 1  # Label relevant hospital admissions with aki

print(f"{(total := df_complete.isna().sum().sum()):_} ({total / df_complete.size:.4f} %) total missing values in dataset;\n"
      f"{(total := df_complete.shape[0]):_} hospitalisations with any measurement; "
      f"{(cases := df_complete.aki.sum())} ({cases / total:.6f} %) AKI-cases;")

# If a proxy was never measured during an hospitalisation, the hospitalisation was not included in the for-loop above
# Thus, no interpolation will take place (also: for interpolation a minimum of 1 value is required)
df_complete.dropna(inplace=True)
print(f"{(total := df_complete.shape[0]):_} hospitalisations with at least one measurement for each proxy;\n"
      f"{(cases := df_complete.aki.sum())} ({cases / total:.6f} %) AKI-cases; Imbalance ratio: {imbalance_ratio(df_complete['aki'].to_numpy())}")
df_complete.to_csv("./data/aki_dataset.csv")
print(f"Shape of dataset: {df_complete.shape}")
