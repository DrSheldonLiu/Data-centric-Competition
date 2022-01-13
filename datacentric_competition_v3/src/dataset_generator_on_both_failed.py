import imageio
import pandas as pd
from tensorflow import keras
import json
import sys
from time import time
import matplotlib.pyplot as plt
import argparse
import os
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from PIL import Image
import tensorflow as tf
from glob import glob
import numpy as np
import os


class MyDatasetGenerator:
    digit_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    label = list(range(len(digit_list)))

    def __init__(self, parent_dir, train_val_flag):
        self.parent_dir = parent_dir
        self.train_val_flag = train_val_flag
        self.png_list = self.get_png_list()
        self.ds = self.generate_dataset()

    def get_png_list(self):
        if self.train_val_flag is not None:
            ds = glob(os.path.join(self.parent_dir, self.train_val_flag, '*', '*.png'))
        else:
            ds = glob(os.path.join(self.parent_dir, '*', '*.png'))

        return ds

    @staticmethod
    def get_label_from_pp(pp):
        label_dict = dict(zip(MyDatasetGenerator.digit_list, MyDatasetGenerator.label))
        return label_dict[os.path.basename(os.path.dirname(pp))]

    def my_gen_with_fname(self, stop):
        i = 0
        while i < stop:
            png_sample = self.png_list[i]
            img_ = tf.image.resize(tf.io.decode_png(tf.io.read_file(png_sample), channels=3), (32, 32))
            label = tf.one_hot(MyDatasetGenerator.get_label_from_pp(png_sample), depth=len(self.digit_list))
            yield img_, label, png_sample
            i += 1

    def generate_dataset(self):
        my_ds = tf.data.Dataset.from_generator(
            self.my_gen_with_fname, args=[len(self.png_list)],
            output_types=(tf.float32, tf.int16, tf.string),
            output_shapes=((32, 32, 3), (len(MyDatasetGenerator.digit_list), ), ()))
        return my_ds


class MyModelLoader:
    def __init__(self, ckpt_path):
        pass
        self.ckpt_path = ckpt_path
        self.model = self.build_model()

    def build_model(self):
        base_model = tf.keras.applications.ResNet50(
            input_shape=(32, 32, 3),
            include_top=False,
            weights=None,
        )
        base_model = tf.keras.Model(
            base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
        )

        inputs = tf.keras.Input(shape=(32, 32, 3))
        x = tf.keras.applications.resnet.preprocess_input(inputs)
        x = base_model(x)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(10, activation='softmax')(x)
        model = tf.keras.Model(inputs, x)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        print(model.summary())
        model.load_weights(os.path.join(self.ckpt_path))
        return model


class MyAnalyser:
    def __init__(self):
        pass


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices('gpu')
    if len(gpus) != 0:
        tf.config.experimental.set_memory_growth(gpus[0], True)

    tar_path = "D:/datacentric_competition_v3/new_start"
    model_name_dirname = "SWL_0409_Codalab"
    now = time()
    tar_log_tsv_fname = tar_path + f"_{model_name_dirname}" + f"_logs.tsv"
    if os.path.exists(tar_log_tsv_fname):
        print(tar_log_tsv_fname + " existed")
        print(f"{time() - now} elapsed")
        exit(0)
    else:
        parent_dir = tar_path
        ds_gen = MyDatasetGenerator(parent_dir=parent_dir, train_val_flag='train')
        # ds_gen = MyDatasetGenerator(parent_dir=parent_dir, train_val_flag=None)
        ds = ds_gen.ds
        ds = ds.batch(batch_size=32)

        model = MyModelLoader(ckpt_path=os.path.join(model_name_dirname, "best_model")).model

        pred_ = list()
        gt_ = list()
        png_list_ = list()
        pred_prob = list()
        len_ds = len(ds_gen.png_list)
        counter = 0
        for x, y, z in ds:
            pred_p = model.predict(x)
            pred_prob.append(pred_p)
            pred = np.argmax(pred_p, axis=1)
            gt = np.argmax(y, axis=1)
            # print(pred, y, z)
            png_list = np.array([i.decode('utf-8') for i in z.numpy()])
            # print(png_list[pred != gt])
            pred_ += list(pred)
            gt_ += list(gt)
            png_list_ += list(png_list)
            counter += 1
            print(f"{counter * 32} out of {len_ds} processed")
            pass
        print('accuracy is ', sum([i == j for i, j in zip(pred_, gt_)]) / len(gt_))

        pred_np, gt_np = np.array(pred_), np.array(gt_)
        png_list_np = np.array(png_list_)
        pred_prob_np = np.array(pred_prob)

        mask = pred_np != gt_np
        failed_png_list = png_list_np[mask]
        failed_pred_list = pred_np[mask]

        failed_prob_np_concat = np.concatenate(pred_prob_np, axis=0).round(2)
        failed_pred_prob_np_filter = failed_prob_np_concat[mask, :]

        failed_gt_list = gt_np[mask]
        label_dict = dict(zip(MyDatasetGenerator.label, MyDatasetGenerator.digit_list))
        failed_pred_list = [label_dict[i] for i in failed_pred_list]
        failed_gt_list = [label_dict[i] for i in failed_gt_list]

        failed_logs = pd.DataFrame(list(zip(failed_png_list, failed_pred_list, failed_gt_list)), columns=['png_name', 'pred', 'gt'])
        failed_prob_df = pd.DataFrame(failed_pred_prob_np_filter, columns=[f"{label_dict[i]}" for i in range(10)])

        failed_logs = pd.concat([failed_logs, failed_prob_df], axis=1)
        failed_logs.to_csv(tar_log_tsv_fname, sep='\t', index=False)
        print(tar_log_tsv_fname + " saved")
        print(f"{time() - now} elapsed")
        # for fname, pp, gg in zip(failed_png_list, failed_pred_list, failed_gt_list):
        #     img = Image.open(fname)
        #     plt.imshow(img, cmap='gray')
        #     plt.title(f'pred: {label_dict[pp]}, gt: {label_dict[gg]}')
        #     plt.show()
