import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

parser = argparse.ArgumentParser(description="Generates tf.dataset based on Path argument.")
parser.add_argument("filepath", metavar="Path", type=str, nargs="?", default="./data/hist", help="The path to the folder containing PNGs.")


@tf.function
def fp_to_resized_image(filepath):
    parts = tf.strings.split(filepath, sep=os.sep)
    filename = parts[-1]
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
    return image, filename


def generate_dataset(filepath: str):
    ds_files = tf.data.Dataset.list_files(filepath)
    ds_images = ds_files.map(fp_to_resized_image).batch(ds_files.cardinality())  # Map preprocessing function to all files in `ds_files`

    X_data, y_data = np.empty(1), np.empty(1)
    for images, labels in ds_images.take(-1):
        X_data = images.numpy()
        y_data = np.vectorize(lambda x: x.decode("utf-8"))(labels.numpy())

    return X_data, y_data


def show(image, label):
    plt.figure()
    plt.imshow(image, vmin=0, vmax=1)
    plt.colorbar()
    plt.title(label)
    plt.show()


if __name__ == "__main__":
    args = parser.parse_args()
    X_data, y_data = generate_dataset(args.filepath + "**/*.png")

    print(X_data.shape, y_data.shape)
    # show(X_data[0], y_data[0])
