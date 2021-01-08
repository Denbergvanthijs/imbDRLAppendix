import argparse
import csv

import numpy as np
from imbDRL.metrics import classification_metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend
from tensorflow.keras.layers import (Concatenate, Conv2D, Dense, Flatten,
                                     Input, MaxPooling2D)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.layers.core import Dropout
from tqdm import tqdm

from histology_preprocessing import (generate_dataset, read_dataframe,
                                     relabel_by_column)

parser = argparse.ArgumentParser(description="Generates tf.dataset based on Path argument.")
parser.add_argument("imagepath", metavar="Path", type=str, nargs="?", default="./data/hist", help="The path to the folder containing PNGs.")
parser.add_argument("csvpath", metavar="Path", type=str, nargs="?", default="./data/AE_20201412.csv", help="The path to the csv-file.")
args = parser.parse_args()

metrics = [Precision(name="precision"), Recall(name="recall")]
X_img, y_img = generate_dataset(args.imagepath)  # X are the images, y are the study numbers
df = read_dataframe(args.csvpath)  # Indexes are the study numbers
df = df[(df.Gender == "1") & (df.Hospital == "2")]
df = df[(df.restenos != -1) & (df.restenos != 2)]
df["month"] = df["dateok"].dt.month
df["dateok"] = df["dateok"].dt.year
print(f"Restenosis:\n{df.restenos.value_counts().to_string()}")

X_structured = list(relabel_by_column(y_img, df[col], default=-1)
                    for col in ["restenos", "Age", "arteryop", "dateok"])  # Keep same order as X_img and y_label
X_structured = np.column_stack(X_structured)  # In the same order as X_img

mask = np.isin(X_structured[:, 0], (0, 1))  # Only keep rows with valid label
X_img = X_img[mask]  # Remove all rows without available label in df
X_structured = X_structured[mask]

_X_train, X_test, _y_train, y_test = train_test_split(
    X_img, X_structured, test_size=0.2, random_state=42)  # Ensure same train/test split every time

X_test_struct = y_test[:, 1:]
y_test = y_test[:, 0]

model1_in = Input(shape=(256, 256, 1))
model1_out = Conv2D(32, kernel_size=(5, 5), activation="relu")(model1_in)
model1_out = MaxPooling2D(pool_size=(2, 2))(model1_out)
model1_out = Conv2D(32, kernel_size=(5, 5), activation="relu")(model1_out)
model1_out = MaxPooling2D(pool_size=(2, 2))(model1_out)
model1_out = Flatten()(model1_out)
model1_out = Dropout(0.5)(model1_out)
model1_out = Dense(256, activation="relu")(model1_out)

model2_in = Input(shape=(3,))
model2_out = Dense(40, activation="relu")(model2_in)
model2_out = Dropout(0.2)(model2_out)
model2_out = Dense(40, activation="relu")(model2_out)
model2_out = Dropout(0.2)(model2_out)

outputs = Concatenate()([model1_out, model2_out])
outputs = Dense(1, activation="sigmoid")(outputs)

thresholds = np.arange(0.0, 1, 0.01)
fp_NN = "./results/histology/nn_dual.csv"
fp_dta = "./results/histology/dta_dual.csv"
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
    X_train_struct = y_train[:, 1:]  # Split structured data from labels
    y_train = y_train[:, 0]
    X_val_struct = y_val[:, 1:]
    y_val = y_val[:, 0]

    model = Model(inputs=[model1_in, model2_in], outputs=outputs)
    model.compile(optimizer=Adam(0.00025), loss="binary_crossentropy", metrics=metrics)

    validation_data = ([X_val, X_val_struct], y_val)
    model.fit([X_train, X_train_struct], y_train, epochs=30, batch_size=32, validation_data=validation_data, verbose=0)

    y_pred_val = model([X_val, X_val_struct], training=False).numpy()  # Faster than .predict() for small batches
    y_pred_test = model([X_test, X_test_struct], training=False).numpy()
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

plot_model(model, to_file="dual_model.png", show_shapes=True)
