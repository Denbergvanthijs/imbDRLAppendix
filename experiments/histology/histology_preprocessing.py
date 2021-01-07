import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

sns.set_palette("colorblind")


@tf.function
def fp_to_resized_image(filepath: str):
    parts = tf.strings.split(filepath, sep=os.sep)
    # Only keep filename, remove AE from AE1234
    filename = tf.strings.substr(tf.strings.split(parts[-1], sep=".", maxsplit=1)[0], pos=2, len=4)
    image = tf.io.read_file(filepath)
    image = tf.image.decode_png(image, channels=4)  # Skip alpha layer
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert range 0-255 to 0-1
    image = 1 - image  # Invert image, empty space and padding now both equal 0
    image = tf.image.resize_with_pad(image, 256, 256)  # Keep original aspect ratio. TODO: determine if keeping asp. rat. is important
    alpha = image[:, :, 3] < 1  # True if not translucent
    image = image[:, :, :3]  # RGBA to RGB
    image = tf.image.rgb_to_grayscale(image)
    image = tf.squeeze(image)  # Reduce last dimension
    image = tf.where(alpha, image, 0)  # If translucent, set pixel-value to 0
    image = tf.reshape(image, image.shape + [1, ])
    return image, filename


def generate_dataset(filepath: str):
    ds_files = tf.data.Dataset.list_files(filepath + "/*.png")
    ds_images = ds_files.map(fp_to_resized_image).batch(ds_files.cardinality())  # Map preprocessing function to all files in `ds_files`

    X_data, y_data = np.empty(1), np.empty(1)
    for images, labels in ds_images.take(-1):
        X_data = images.numpy()
        y_data = np.vectorize(lambda x: x.decode("utf-8"))(labels.numpy())  # Decode TF bytes to python strings

    y_data = y_data.astype(np.int32)
    return X_data, y_data


def read_dataframe(filepath: str):
    dtypes = {"STUDY_NUMBER": int, "Age": int, "Gender": "category",
              "Hospital": "category", "dateok": str, "arteryop": int, "restenos": str}
    df = pd.read_csv(filepath, dtype=dtypes, parse_dates=["dateok"], index_col="STUDY_NUMBER")
    df.drop(columns=["Unnamed: 0"], inplace=True)  # Drop old index-col from original csv-file
    df["restenos"].fillna(-1.0, inplace=True)  # Fill NaNs, it is up to the user to determine the fate of missing values
    df["restenos"] = df["restenos"].astype(int)
    arteryop_categories = np.arange(df["arteryop"].min(), df["arteryop"].max() + 1)
    df["arteryop"] = pd.Categorical(df["arteryop"], categories=arteryop_categories, ordered=True)  # For plotting the data ordered
    return df


def relabel_by_column(y_data: np.ndarray, column: pd.Series, default: int = -1):
    """
    Relabels `y_data` array by looking for corresponding value in `column` Series.
    If value at `y_data` is not available in `column`, a default value will be chosen.
    Values in `y_data` must correspond to index of `column`.
    """
    d = column.to_dict()
    _y_data = y_data.copy()
    for c, value in enumerate(_y_data):
        _y_data[c] = d.get(value, default)
    return _y_data


def show(image, label):
    plt.figure()
    plt.imshow(image, vmin=0, vmax=1, cmap="Greys")
    plt.colorbar()
    plt.title(label)
    plt.grid(False)
    plt.show()


def data_exploration(df):
    print(df.head())
    print(df.describe(include="all"))
    print(f"Age: {df.Age.mean():.2f}, ± {df.Age.std():.2f}\n"
          f"Gender:\n{df.Gender.value_counts().to_string()}\n"
          f"Restenos:\n{df.restenos.value_counts().to_string()}\n"
          f"Hospital:\n{df.Hospital.value_counts().to_string()}")

    # Distributions
    _, axis = plt.subplots(1, 2)
    for c, col in enumerate(["arteryop", "dateok"]):
        sns.histplot(data=df, x=col, ax=axis[c])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generates tf.dataset based on Path argument.")
    parser.add_argument("imagepath", metavar="Path", type=str, nargs="?",
                        default="./data/hist", help="The path to the folder containing PNGs.")
    parser.add_argument("csvpath", metavar="Path", type=str, nargs="?", default="./data/AE_20201412.csv", help="The path to the csv-file.")

    args = parser.parse_args()
    X_data, y_data = generate_dataset(args.imagepath)

    print(X_data.shape, y_data.shape)
    print(X_data.dtype, y_data.dtype)

    df = read_dataframe(args.csvpath)
    df = df[(df.Gender == "1") & (df.Hospital == "2")]
    df = df[(df.restenos != -1) & (df.restenos != 2)]
    print(df.restenos.value_counts())
    # data_exploration(df)
    y_data_labeled = relabel_by_column(y_data, df["restenos"], default=-1)
    print(f"Counter: {dict(zip(*np.unique(y_data_labeled, return_counts=True)))}")
    # show(X_data[0], f"Study Number: {y_data[0]}; Restenosis: {y_data_labeled[0]}")
