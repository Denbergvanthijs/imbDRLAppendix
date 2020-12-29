import pandas as pd

experiment = "creditcardfraud"
usecols = ("F1", "Precision", "Recall")
rounding = 3

df = pd.read_csv(f"./results/{experiment}/nn.csv", usecols=usecols)
print(pd.DataFrame([df.mean(), df.std()]).T.round(rounding))

df = pd.read_csv(f"./results/{experiment}/dta.csv", usecols=usecols)
print(pd.DataFrame([df.mean(), df.std()]).T.round(rounding))

df = pd.read_csv(f"./results/{experiment}/dqn.csv", usecols=usecols)
print(pd.DataFrame([df.mean(), df.std()]).T.round(rounding))

# df = pd.read_csv(f"./results/histology/dqn.csv", usecols=["Gmean", "P"])
# print(df.groupby("P").mean().round(rounding))

# df = pd.read_csv("./results/histology/dqn.csv")
# print(pd.DataFrame([df.mean(), df.std()]).T.round(rounding))
# print(df)
