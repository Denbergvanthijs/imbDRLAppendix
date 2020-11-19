import csv

import numpy as np
from imbDRL.data import get_train_test_val, load_creditcard
from imbDRL.metrics import classification_metrics
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.optimizers import Adam
from tqdm import tqdm

min_class = [1]  # Labels of the minority classes
maj_class = [0]  # Labels of the majority classes
X_train, y_train, X_test, y_test = load_creditcard(normalization=True, fp_train="./data/credit0.csv", fp_test="./data/credit1.csv")
metrics = [Precision(name="precision"), Recall(name="recall")]
thresholds = np.arange(0, 1, 0.01)

fp_baseline = "./results/creditcardfraud/baseline.csv"
fp_dta = "./results/creditcardfraud/dta.csv"
fieldnames = ("Gmean", "F1", "Precision", "Recall", "TP", "TN", "FP", "FN")

# Create empty files
with open(fp_baseline, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
with open(fp_dta, "w", newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

# Run the model ten times
for _ in tqdm(range(10)):
    # New train-test split
    X_train, y_train, X_test, y_test, X_val, y_val = get_train_test_val(
        X_train, y_train, X_test, y_test, min_class, maj_class, val_frac=0.2, print_stats=False)

    model = Sequential([Dense(256, activation="relu", input_shape=(X_train.shape[-1],)),
                        Dropout(0.2),
                        Dense(256, activation="relu"),
                        Dropout(0.2),
                        Dense(1, activation="sigmoid")])
    model.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=metrics)
    model.fit(X_train, y_train, epochs=30, batch_size=2048, validation_data=(X_val, y_val), verbose=0)

    # Predictions of model for `X_test`
    y_pred = model(X_test).numpy()
    baseline_stats = classification_metrics(y_test, np.around(y_pred).astype(int))

    # Write current baseline run to `fp_baseline`
    with open(fp_baseline, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(baseline_stats)

    # F1 of every threshold
    f1scores = [classification_metrics(y_test, (y_pred >= th).astype(int)).get("F1") for th in thresholds]
    # Select threshold with highest F1
    dta_stats = classification_metrics(y_test, (y_pred >= thresholds[np.argmax(f1scores)]).astype(int))

    # Write current DTA run to `fp_dta`
    with open(fp_dta, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(dta_stats)
