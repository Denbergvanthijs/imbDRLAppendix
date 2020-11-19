import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(context="paper")
sns.set_palette("colorblind")

df_dqn = pd.read_csv("./results/creditcardfraud/dqn.csv")
df_dqn["Methode"] = "DQN"
df_baseline = pd.read_csv("./results/creditcardfraud/baseline.csv")
df_baseline["Methode"] = "Baseline"
df_dta = pd.read_csv("./results/creditcardfraud/dta.csv")
df_dta["Methode"] = "DTA"
df_all = pd.concat((df_dqn, df_baseline, df_dta))

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))

sns.barplot(x="Methode", y="F1", data=df_all, ax=ax1)
sns.barplot(x="Methode", y="Precision", data=df_all, ax=ax2)
sns.barplot(x="Methode", y="Recall", data=df_all, ax=ax3)

fig.suptitle("F1-score, Precision en Recall voor het DQN-algoritme, de baseline\n"
             "en de DTA-methode op de creditcard-fraude dataset.")
plt.setp((ax1, ax2, ax3), ylim=(0.0, 1))
plt.setp((ax1, ax2, ax3), xlabel="")
plt.savefig("./plots/creditcardfraud.png", dpi=300)
plt.show()
