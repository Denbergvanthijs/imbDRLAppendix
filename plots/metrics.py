import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

sns.set(context="paper")
sns.set_palette("colorblind")

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(12, 3), sharey=True)
P = 0.5  # Chance of choosing positive class
n_neg = 1_000  # Number of negative elements
negatives = np.zeros(n_neg)  # Const list of True Negatives
acc, spec, recall, prec, f1, imb_ratio = [], [], [], [], [], []

for n_pos in np.arange(1, n_neg + 1, 10):
    y_true = np.concatenate((np.ones(n_pos), negatives))  # Ground Truth
    y_pred = np.random.choice(2, size=n_pos + n_neg, p=[1 - P, P])  # Choose 0 with p=1-P or 1 with p=P

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc.append((tp + tn) / (tp + tn + fn + fp))
    spec.append(tn / (tn + fp))
    recall.append(tp / (tp + fn))
    prec.append(tp / (tp + fp))
    f1.append(2 * tp / (2 * tp + fp + fn))

    imb_ratio.append(n_pos / n_neg)

sns.lineplot(x=imb_ratio, y=acc, lw=1, ax=ax1)
sns.lineplot(x=imb_ratio, y=spec, lw=1, ax=ax2)
sns.lineplot(x=imb_ratio, y=recall, lw=1, ax=ax3)
sns.lineplot(x=imb_ratio, y=prec, lw=1, ax=ax4)
sns.lineplot(x=imb_ratio, y=f1, lw=1, ax=ax5)

fig.suptitle("Verandering van imbalance ratio en gevolg voor verschillende metriek.\n"
             f"Artificiële dataset met {n_neg} waarden voor negatieve klasse en "
             f"1 tot en met {n_neg} waarden voor positieve klasse.\n"
             f"Voorspellingen met kans {P} voor de minderheidsklasse.")

for ax, title in zip((ax1, ax2, ax3, ax4, ax5), ("Nauwkeurigheid", "Specificiteit", "Sensitiviteit/Recall", "Precision", "F1-score")):
    ax.set_title(title)

plt.subplots_adjust(top=0.80)
ax1.set_ylabel("Score van metriek")
ax3.set_xlabel("Imbalance ratio")
plt.setp((ax1, ax2, ax3, ax4, ax5), xlim=(0, 1))
plt.setp((ax1, ax2, ax3, ax4, ax5), ylim=(0, 1))
plt.setp((ax1, ax2, ax3, ax4, ax5), aspect=1)
plt.savefig("./plots/metrics.png", dpi=300)
plt.show()
