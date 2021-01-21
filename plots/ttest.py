import pandas as pd
from scipy.stats import ttest_ind

experiments = ("creditcardfraud", "histology", "aki")
usecols = ("F1", )  # "Precision", "Recall"
alpha = 0.05

for experiment in experiments:
    df_nn = pd.read_csv(f"./results/{experiment}/nn.csv", usecols=usecols)
    df_dqn = pd.read_csv(f"./results/{experiment}/dqn.csv", usecols=usecols)

    print(experiment)
    for col in usecols:
        _, p = ttest_ind(df_nn[col], df_dqn[col], equal_var=False, alternative="greater")
        # Welch’s t-test

        if (1 - p) > alpha:
            print(f"{col:>12} p: {1-p:.3f}; Accept H0; Same performance;")
        else:
            print(f"{col:>12} p: {1-p:.3f}; Reject H0; Better performance;")
