from glob import glob
import os
import cv2
import numpy as np
import shutil
import random
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from scipy import ndimage as ndi
import functools
from random import shuffle
import string


class GridOnTheWay:
    """ do batch augmentation on data-centric dataset """
    digits_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    tar_row, tar_col = 600, 600

    def __init__(self, parent_dir, tar_ext="_griding", random_name=True):
        self.parent_dir = parent_dir
        self.src_imgs, self.src_images_path_dict = self._set_up()
        self.tar_ext = tar_ext
        self.name_ext = "-ce5d-11eb-b317-38f9d35ea60f"
        self.random_name = random_name

    def _set_up(self):
        src_images = glob(os.path.join(self.parent_dir, '*', '*.png'))
        src_images_path_dict = {}
        for i in self.digits_list:
            src_images_path_dict[i] = glob(os.path.join(self.parent_dir, i, '*.png'))
        return src_images, src_images_path_dict

    def __call__(self, *args, **kwargs):
        for kk, vv in self.src_images_path_dict.items():
            print(f'processing {kk}')
            list(map(self.grid_it_up_left, vv[:10]))
            break

    @staticmethod
    def get_random_name():
        return random.choices([i for i in string.ascii_lowercase], k=1)[0] + \
               ''.join(random.choices(''.join(list(map(str, list(range(10))))) + string.ascii_lowercase, k=7))

    def grid_it_up_left(self, src_img):
        print(f"processing left corner griding -> {src_img}")
        img = Image.open(src_img)
        img = ImageOps.grayscale(img)
        img_array = np.array(img)
        h, w = img_array.shape[0], img_array.shape[1]
        img_padded = np.ones((h * 2, w * 2)) * 255
        img_padded[:h, :w] = img_array
        plt.figure()
        plt.imshow(img_padded, cmap='gray')
        plt.show()
        img_aug = Image.fromarray(img_padded, "L")
        img_aug = img
        # img_aug = img_aug.resize((600, 600))
        tar_img_name = GridOnTheWay.get_random_name() + self.name_ext + ".png"
        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(
                os.path.basename(self.parent_dir),
                os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))
        img_aug.save(tar_img_path)


if __name__ == "__main__":
    parent_dir = r"D:\tars\iter3_002\train"
    gow = GridOnTheWay(parent_dir=parent_dir)
    gow()