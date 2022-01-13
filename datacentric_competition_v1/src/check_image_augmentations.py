import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from datetime import datetime
import argparse
import os
import numpy as np
import json
import sys

directory = "../data_cleaned"
user_data = valid_data = directory
test_data = "./label_book"

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8 * 4
tf.random.set_seed(123)

train = tf.keras.preprocessing.image_dataset_from_directory(
    user_data + '/train',
    labels="inferred",
    label_mode="categorical",
    class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
    shuffle=False,
    seed=123,
    batch_size=batch_size,
    image_size=(32, 32),
)

fig, axes = plt.subplots(4, 8)
for idx, (image, label) in enumerate(train.take(1)):
    image_np = image.numpy()
    print(image_np.shape)
    for i in range(image_np.shape[0]):
        axes[i // 8, i % 8].imshow(image_np[i, ...].astype(np.uint8))
        axes[i // 8, i % 8].title.set_text(
            ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][np.argmax(label[i, ...], axis=-1)])
    #plt.show()

#train = train.map(lambda image, label: (tf.keras.layers.GaussianNoise(1)(image), label))
#train = train.map(lambda image, label: (tf.image.random_crop(image, 0.7 * image.shape[1:]), label))

fig, axes = plt.subplots(4, 8)
for idx, (image, label) in enumerate(train.take(1)):
    image_np = image.numpy()
    print(image_np.shape)
    for i in range(image_np.shape[0]):
        print(image_np[i, ...].shape)
        axes[i // 8, i % 8].imshow(tf.image.random_crop(image_np[i, ...].astype(np.uint8), (26, 26, 3)))
        axes[i // 8, i % 8].title.set_text(
            ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"][np.argmax(label[i, ...], axis=-1)])
    plt.show()
