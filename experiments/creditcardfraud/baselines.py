import numpy as np
from imbDRL.data import load_csv
from imbDRL.metrics import classification_metrics
from sklearn.dummy import DummyClassifier

_, _, X_test, y_test = load_csv("./data/credit0.csv", "./data/credit1.csv", "Class", ["Time"], normalization=True)

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
for _ in range(1000):
    stats_uniform.append(uniform_baseline(X_test, y_test))
    stats_minority.append(minority_baseline(X_test, y_test))

for lst in stats_uniform, stats_minority:
    for metric in "F1", "Precision", "Recall":
        print(f"{metric}: {np.round(np.mean([d[metric] for d in lst]), 3)} ± {np.round(np.std([d[metric] for d in lst]), 3)}")
    print()
