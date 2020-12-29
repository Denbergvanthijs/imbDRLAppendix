import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D

sns.set(context="paper")
sns.set_palette("colorblind")


def plot_results(dataset: str, title: str):
    df_NN = pd.read_csv(f"./results/{dataset}/nn.csv")
    df_NN["Methode"] = "NN"
    df_dta = pd.read_csv(f"./results/{dataset}/dta.csv")
    df_dta["Methode"] = "DTA"
    df_dqn = pd.read_csv(f"./results/{dataset}/dqn.csv")
    df_dqn["Methode"] = "DQN"
    df_all = pd.concat((df_NN, df_dta, df_dqn))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, tight_layout=True, figsize=(10, 4))

    sns.barplot(x="Methode", y="F1", data=df_all, ax=ax1)
    sns.barplot(x="Methode", y="Precision", data=df_all, ax=ax2)
    sns.barplot(x="Methode", y="Recall", data=df_all, ax=ax3)

    fig.suptitle("F1-score, Precision en Recall voor het standaard NN, de DTA-methode\n"
                 f"en het DQN-algoritme {title}.")
    plt.setp((ax1, ax2, ax3), ylim=(0.0, 1))
    plt.setp((ax1, ax2, ax3), xlabel="")
    plt.savefig(f"./plots/{dataset}.png", dpi=300)
    plt.legend(handles=[Line2D([], [], color="black", label="95% conf. interval")])
    plt.show()


datasets = ("creditcardfraud", "histology", "aki")
titles = ("op de Creditcard-fraude dataset", "op de histologie dataset van het UMCU", "op de MIMIC-IV AKI dataset")
for ds, title in zip(datasets, titles):
    plot_results(ds, title)
