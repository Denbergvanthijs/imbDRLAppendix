import argparse
import csv

import numpy as np
from imbDRL.metrics import classification_metrics
from imbDRL.utils import imbalance_ratio
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers.core import Dropout
from tqdm import tqdm

from histology_preprocessing import read_dataframe

parser = argparse.ArgumentParser(description="Generates dataset based on Path argument.")
parser.add_argument("imagepath", metavar="Path", type=str, nargs="?", default="./data/hist", help="The path to the folder containing PNGs.")
parser.add_argument("csvpath", metavar="Path", type=str, nargs="?", default="./data/AE_20201412.csv", help="The path to the csv-file.")
args = parser.parse_args()

metrics = [Precision(name="precision"), Recall(name="recall")]
df = read_dataframe(args.csvpath)
df = df[(df.Gender == "1") & (df.Hospital == "2")]
df = df[(df.restenos != -1) & (df.restenos != 2)]
y = df["restenos"].to_numpy()
print(f"Imbalance ratio: {imbalance_ratio(y):.4f}\nRestenos:\n{df['restenos'].value_counts().to_string()}\n")

df.drop(columns=["restenos", "Gender", "Hospital"], inplace=True)
df["month"] = df["dateok"].dt.month
df["dateok"] = df["dateok"].dt.year
df = df.reset_index(drop=True)  # Drop study number
df = df.astype("int32")
df = (df - df.min()) / (df.max() - df.min())  # Normalization
# print(f"{df.sample(3)}\n")

# Ensure same train/test split every time
_X_train, X_test, _y_train, y_test = train_test_split(df[["Age", "arteryop"]].to_numpy(), y, test_size=0.2, random_state=42)

thresholds = np.arange(0.0, 1, 0.01)
fp_NN = "./results/histology/nn_struct.csv"
fp_dta = "./results/histology/dta_struct.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN")

# Create empty files
with open(fp_NN, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
with open(fp_dta, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run the model ten times
for _ in tqdm(range(10)):
    # New train-test split
    X_train, X_val, y_train, y_val = train_test_split(_X_train, _y_train, test_size=0.2)  # 64/20/16 split

    backend.clear_session()
    model = Sequential([Input(shape=(2,)),
                        Dense(40, activation="relu"),
                        Dropout(0.2),
                        Dense(40, activation="relu"),
                        Dropout(0.2),
                        Dense(1, activation="sigmoid")])
    model.compile(optimizer=Adam(0.00025), loss="binary_crossentropy", metrics=metrics)
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    y_pred_val = model(X_val, training=False).numpy()
    y_pred_test = model(X_test, training=False).numpy()
    NN_stats = classification_metrics(y_test, np.around(y_pred_test).astype(int))

    # Write current NN run to `fp_NN`
    with open(fp_NN, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(NN_stats)

    # Validation F1 of every threshold
    f1scores = [classification_metrics(y_val, (y_pred_val >= th).astype(int)).get("F1") for th in thresholds]

    # Select threshold with highest validation F1
    dta_stats = classification_metrics(y_test, (y_pred_test >= thresholds[np.argmax(f1scores)]).astype(int))

    # Write current DTA run to `fp_dta`
    with open(fp_dta, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dta_stats)
