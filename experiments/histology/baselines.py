import argparse

import numpy as np
from imbDRL.metrics import classification_metrics
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split

from histology_preprocessing import (generate_dataset, read_dataframe,
                                     relabel_by_column)

parser = argparse.ArgumentParser(description="Generates tf.dataset based on Path argument.")
parser.add_argument("imagepath", metavar="Path", type=str, nargs="?", default="./data/hist", help="The path to the folder containing PNGs.")
parser.add_argument("csvpath", metavar="Path", type=str, nargs="?", default="./data/AE_20201412.csv", help="The path to the csv-file.")
args = parser.parse_args()

X, y = generate_dataset(args.imagepath)
df = read_dataframe(args.csvpath)
# df = df[df.Hospital == "2"]
df = df[df.Gender == "1"]
# df = df[df.dateok.dt.year >= 2010]
print(f"Restenosis:\n{df.restenos.value_counts().to_string()}")

y = relabel_by_column(y, df["restenos"], default=-1)
y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Ensure same train/test split every time


def uniform_baseline(X, y):
    uniform_clf = DummyClassifier(strategy="uniform")
    uniform_clf.fit(X, y)
    y_pred = uniform_clf.predict(X)
    return classification_metrics(y, y_pred)


def minority_baseline(X, y):
    most_frequent_clf = DummyClassifier(strategy="constant", constant=1)
    most_frequent_clf.fit(X, y)
    y_pred = most_frequent_clf.predict(X)
    return classification_metrics(y, y_pred)


stats_uniform = []
stats_minority = []
for _ in range(100):
    stats_uniform.append(uniform_baseline(X_test, y_test))
    stats_minority.append(minority_baseline(X_test, y_test))

for l in stats_uniform, stats_minority:
    for metric in "F1", "Precision", "Recall":
        print(f"{metric}: {np.round(np.mean([d[metric] for d in l]), 3)} ± {np.round(np.std([d[metric] for d in l]), 3)}")
    print()
