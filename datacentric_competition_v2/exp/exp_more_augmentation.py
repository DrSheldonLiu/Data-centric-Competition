from glob import glob
import os
import cv2
from PIL import Image, ImageOps
from random import shuffle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

font = {'family': 'normal',
        'weight': 'bold',
        'size': 5}

matplotlib.rc('font', **font)
parent_dir = "D:/datacentric_competition_v2/data/train"


def aug_fliplr(img):
    return ImageOps.mirror(img)


def aug_fliph(img):
    return ImageOps.flip(img)


def aug_crop(img):
    img_ = 255 - np.array(img)
    mask = (img_ != 0)
    x_min, y_min = np.argwhere(mask).min(axis=0)
    x_max, y_max = np.argwhere(mask).max(axis=0)
    return Image.fromarray(255 - img_[x_min: x_max, y_min: y_max], mode="L")


def aug_noise(img):
    img_ = 255 - np.array(img)
    noise = np.random.rand(*(img_.shape)) * 255
    img_aug = img_ + noise
    return Image.fromarray(255 - img_aug)


def aug_add_dot_lines(img):
    img_ = np.array(img)
    flag = np.random.randint(2)

    random_count = np.random.randint(10) + 2

    random_row = np.random.randint(img_.shape[0])
    random_col = np.random.randint(img_.shape[1])
    start_point = 30
    random_row = int(random_row/img_.shape[0] * (img_.shape[0] - start_point * 2) + start_point)
    random_col = int(random_col/img_.shape[1] * (img_.shape[1] - start_point * 2) + start_point)
    if flag == 1:
        for idx in range(random_count):
            thickness = np.random.randint(low=3, high=10)
            line_line_distance = np.random.randint(low=5, high=10)
            img_[(random_row + idx * line_line_distance): (random_row + idx * line_line_distance + thickness), ::3] = 0
    else:
        for idx in range(random_count):
            thickness = np.random.randint(low=3, high=10)
            line_line_distance = np.random.randint(low=5, high=10)
            img_[::3, (random_col + idx * line_line_distance): (random_col + idx * line_line_distance + thickness)] = 0
    return Image.fromarray(img_)


def exp_i():
    # digit i
    digit_i_list = glob(os.path.join(parent_dir, 'i', '*f.png'))
    shuffle(digit_i_list)

    for gg in [digit_i_list[i: i + 5] for i in range(0, len(digit_i_list), 5)]:
        fig, axes = plt.subplots(5, 6, sharex='row')
        for idx, item in enumerate(gg):
            i_one = item
            # src_img = cv2.imread(i_one, cv2.IMREAD_GRAYSCALE)
            src_img = Image.open(i_one)
            aug_fliplr_img = aug_fliplr(src_img)
            aug_fliph_img = aug_fliph(src_img)
            aug_crop_img = aug_crop(src_img)
            aug_noise_img = aug_noise(src_img)
            aug_dot_line_img = aug_add_dot_lines(src_img)

            # start doing augmentation
            axes[idx, 0].imshow(src_img, cmap='gray')
            axes[idx, 0].set_ylabel(os.path.basename(i_one))
            axes[idx, 1].imshow(aug_fliplr_img, cmap='gray')
            axes[idx, 2].imshow(aug_fliph_img, cmap='gray')
            axes[idx, 3].imshow(aug_crop_img, cmap='gray')
            axes[idx, 4].imshow(aug_noise_img, cmap='gray')
            axes[idx, 5].imshow(aug_dot_line_img, cmap='gray')

        plt.show()


if __name__ == "__main__":
    exp_i()

