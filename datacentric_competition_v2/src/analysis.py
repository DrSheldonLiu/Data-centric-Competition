import numpy as np
import tensorflow as tf
import os
from glob import glob


class AnalysisDataset:
    def __init__(self, parent_dir):
        """
        :param parent_dir: parent_dir contains digits folders
        """
        self.parent_dir = parent_dir
        self.ds = self._get_dataset()

    def _get_dataset(self):
        images = glob(os.path.join(self.parent_dir, '*', '*f.png'))
        labels = list(map(self.parse_img_path, images))
        ds = tf.data.Dataset.from_tensor_slices((images, labels))
        return ds

    @staticmethod
    def parse_img_path(img_p):
        return os.path.basename(os.path.dirname(img_p))

    @staticmethod
    def load_imgs(img_p):
        img = tf.io.read_file(img_p)
        img = tf.io.decode_image(img, channels=3)
        return tf.image.resize(img, [32, 32])

    @staticmethod
    def load_and_convert(x):
        return tf.convert_to_tensor(np.stack([tf.io.decode_image(tf.io.read_file(i.decode("utf-8")), channels=3) for i in x.numpy()], axis=0))


if __name__ == "__main__":
    parent_dir = "D:/datacentric_competition_v2/data/train"
    analysis_ds = AnalysisDataset(parent_dir=parent_dir)
    ds_ = analysis_ds.ds
    ds_ = ds_.batch(batch_size=16)
    for x, y in ds_.take(1):
        print(analysis_ds.load_and_convert(x), y)