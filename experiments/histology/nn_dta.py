import argparse
import csv

import numpy as np
from imbDRL.data import get_train_test_val
from imbDRL.metrics import classification_metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential, backend
from tensorflow.keras.layers import (Conv2D, Dense, Dropout, Flatten,
                                     MaxPooling2D)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

from histology_preprocessing import (generate_dataset, read_dataframe,
                                     relabel_by_column)

parser = argparse.ArgumentParser(description="Generates tf.dataset based on Path argument.")
parser.add_argument("imagepath", metavar="Path", type=str, nargs="?", default="./data/hist", help="The path to the folder containing PNGs.")
parser.add_argument("csvpath", metavar="Path", type=str, nargs="?", default="./data/AE_20201412.csv", help="The path to the csv-file.")
args = parser.parse_args()

X, y = generate_dataset(args.imagepath)  # X are the images, y are the study numbers
df = read_dataframe(args.csvpath)
df = df[(df.Gender == "1") & (df.Hospital == "2")]
df = df[(df.restenos != -1) & (df.restenos != 2)]
print(f"Restenosis:\n{df.restenos.value_counts().to_string()}")

y = relabel_by_column(y, df["restenos"], default=-1)  # Convert study numbers to restenos labels
# y = np.random.choice(2, size=30).astype(np.int32)  # Mock data for testing
# X = np.concatenate([X, X, X])
_X_train, _X_test, _y_train, _y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Ensure same train/test split every time

min_class = [1]  # Labels of the minority classes
maj_class = [0]  # Labels of the majority classes
metrics = [Precision(name="precision"), Recall(name="recall")]

# Thresholds < 0.5 will result in higher recall than standard NN
# Thresholds > 0.5 will result in higher precision than standard NN
thresholds = np.arange(0.0, 1, 0.01)

fp_NN = "./results/histology/nn.csv"
fp_dta = "./results/histology/dta.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN")

# Create empty files
with open(fp_NN, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
with open(fp_dta, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run the model ten times
for _ in tqdm(range(10)):
    # New train-test split
    X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(_X_train, _y_train, _X_test, _y_test, min_class, maj_class,
                                                                        val_frac=0.2, print_stats=False)
    backend.clear_session()
    model = Sequential([Conv2D(32, kernel_size=(5, 5), activation="relu"),
                        MaxPooling2D(pool_size=(2, 2)),
                        Conv2D(32, kernel_size=(5, 5), activation="relu"),
                        MaxPooling2D(pool_size=(2, 2)),
                        Flatten(),
                        Dense(256, activation="relu"),
                        Dropout(0.2),
                        Dense(1, activation="sigmoid")])
    model.compile(optimizer=Adam(0.00025), loss="binary_crossentropy", metrics=metrics)
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Predictions of model for `X_test`
    y_pred_val = model(X_val).numpy()
    y_pred_test = model(X_test).numpy()
    NN_stats = classification_metrics(y_test, np.around(y_pred_test).astype(int))

    # Write current NN run to `fp_NN`
    with open(fp_NN, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(NN_stats)

    # Validation F1 of every threshold
    f1scores = [classification_metrics(y_val, (y_pred_val >= th).astype(int)).get("F1") for th in thresholds]

    # Select threshold with highest validation F1
    dta_stats = classification_metrics(y_test, (y_pred_test >= thresholds[np.argmax(f1scores)]).astype(int))

    # Write current DTA run to `fp_dta`
    with open(fp_dta, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dta_stats)
