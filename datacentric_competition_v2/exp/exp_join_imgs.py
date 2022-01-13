from glob import glob
import os
import cv2
from random import shuffle
import matplotlib.pyplot as plt
import numpy as np


parent_dir = "D:/datacentric_competition_v2/data/train"
#digit_list = [i for i in os.listdir(parent_dir)]


def exp_i():
    # digit i
    digit_i_list = glob(os.path.join(parent_dir, 'i', '*f.png'))
    shuffle(digit_i_list)

    for gg in [digit_i_list[i: i + 25] for i in range(0, len(digit_i_list), 25)]:
        fig, axes = plt.subplots(5, 5)
        for idx, item in enumerate(gg):
            i_one = item
            img_one = cv2.imread(i_one, cv2.IMREAD_GRAYSCALE)
            #merged = np.zeros(img_one.shape)
            #img_one_rolled = np.roll(img_one, 50, axis=1)
            #merged = img_one + img_one_rolled
            #merged = img_one_rolled
            merged = 255 - cv2.dilate(255 - img_one, kernel=np.ones((3, 3), dtype=np.int8), iterations=3)
            # merged[img_one > img_one_rolled] = img_one[img_one > img_one_rolled]
            # merged[img_one <= img_one_rolled] = img_one_rolled[img_one <= img_one_rolled]
            # merged = img_one + np.roll(img_one, 2, axis=1)
            # merged = merged // 43 * 43
            # merged[merged > 130] = 255
            axes[idx // 5, idx % 5].imshow(merged, cmap='gray')
        plt.show()


def exp_ii():
    # digit i
    digit_i_list = glob(os.path.join(parent_dir, 'i', '*f.png'))
    shuffle(digit_i_list)

    for gg in [digit_i_list[i: i + 25] for i in range(0, len(digit_i_list), 25)]:
        fig, axes = plt.subplots(5, 5)
        for idx, item in enumerate(gg):
            i_one = item
            img_one = cv2.imread(i_one, cv2.IMREAD_GRAYSCALE)
            merged = np.zeros(img_one.shape)
            img_one_rolled = np.roll(img_one, 30, axis=1)
            merged[img_one < img_one_rolled] = img_one[img_one < img_one_rolled]
            merged[img_one >= img_one_rolled] = img_one_rolled[img_one >= img_one_rolled]
            # merged = img_one + np.roll(img_one, 2, axis=1)
            # merged = merged // 43 * 43
            # merged[merged > 130] = 255
            axes[idx // 5, idx % 5].imshow(merged, cmap='gray')
        plt.show()


def exp_iii():
    # digit i
    digit_i_list = glob(os.path.join(parent_dir, 'i', '*f.png'))
    shuffle(digit_i_list)
    for gg in [digit_i_list[i: i + 25] for i in range(0, len(digit_i_list), 25)]:
        fig, axes = plt.subplots(5, 5)
        for idx, item in enumerate(gg):
            i_one = item
            img_one = cv2.imread(i_one, cv2.IMREAD_GRAYSCALE)
            img_aug = np.zeros(img_one.shape)
            img_one_rolled = np.roll(img_one, 30, axis=1)
            img_aug[img_one < img_one_rolled] = img_one[img_one < img_one_rolled]
            img_aug[img_one >= img_one_rolled] = img_one_rolled[img_one >= img_one_rolled]

            img_one_rolled_v2 = np.roll(img_one, -30, axis=1)
            img_aug_v2 = np.zeros(img_one.shape)
            img_aug_v2[img_one_rolled_v2 < img_aug] = img_one_rolled_v2[img_one_rolled_v2 < img_aug]
            img_aug_v2[img_one_rolled_v2 >= img_aug] = img_aug[img_one_rolled_v2 >= img_aug]

            # merged = img_one + np.roll(img_one, 2, axis=1)
            # merged = merged // 43 * 43
            # merged[merged > 130] = 255
            axes[idx // 5, idx % 5].imshow(img_aug_v2, cmap='gray')
        plt.show()


def exp_iv():
    # digit i and digit v
    digit_i_list = glob(os.path.join(parent_dir, 'i', '*f.png'))
    digit_v_list = glob(os.path.join(parent_dir, 'v', '*f.png'))
    min_len = min(len(digit_i_list), len(digit_v_list))
    digit_i_list = digit_i_list[:min_len]
    digit_v_list = digit_v_list[:min_len]

    for gg in [list(zip(digit_i_list, digit_v_list))[i: i + 25] for i in range(0, min_len, 25)]:
        fig, axes = plt.subplots(5, 5)
        fig1, axes1 = plt.subplots(5, 5)
        for idx, (i, v) in enumerate(gg):
            img_i = cv2.resize(cv2.imread(i, cv2.IMREAD_GRAYSCALE), (16, 32))
            img_v = cv2.resize(cv2.imread(v, cv2.IMREAD_GRAYSCALE), (16, 32))
            axes[idx // 5, idx % 5].imshow(np.concatenate([np.roll(img_i, 50, axis=1), img_v], axis=1), cmap='gray')
            axes1[idx // 5, idx % 5].imshow(np.concatenate([img_v, np.roll(img_i, -50, axis=1)], axis=1), cmap='gray')
        plt.show()


def exp_vii():
    digit_ii_list = glob(os.path.join(parent_dir, 'ii', '*f.png'))
    digit_v_list = glob(os.path.join(parent_dir, 'v', '*f.png'))
    min_len = min(len(digit_ii_list), len(digit_v_list))
    digit_ii_list = digit_ii_list[:min_len]
    digit_v_list = digit_v_list[:min_len]

    for gg in [list(zip(digit_ii_list, digit_v_list))[i: i + 25] for i in range(0, min_len, 25)]:
        fig, axes = plt.subplots(5, 5)
        fig1, axes1 = plt.subplots(5, 5)
        for idx, (i, v) in enumerate(gg):
            img_ii = cv2.resize(cv2.imread(i, cv2.IMREAD_GRAYSCALE), (16, 32))
            img_v = cv2.resize(cv2.imread(v, cv2.IMREAD_GRAYSCALE), (16, 32))
            axes1[idx // 5, idx % 5].imshow(np.concatenate([img_v, np.roll(img_ii, -50, axis=1)], axis=1), cmap='gray')
        plt.show()
    pass


def exp_viii():
    digit_iii_list = glob(os.path.join(parent_dir, 'iii', '*f.png'))
    digit_v_list = glob(os.path.join(parent_dir, 'v', '*f.png'))
    min_len = min(len(digit_iii_list), len(digit_v_list))
    digit_iii_list = digit_iii_list[:min_len]
    digit_v_list = digit_v_list[:min_len]

    for gg in [list(zip(digit_iii_list, digit_v_list))[i: i + 25] for i in range(0, min_len, 25)]:
        fig, axes = plt.subplots(5, 5)
        fig1, axes1 = plt.subplots(5, 5)
        for idx, (i, v) in enumerate(gg):
            img_iii = cv2.resize(cv2.imread(i, cv2.IMREAD_GRAYSCALE), (16, 32))
            img_v = cv2.resize(cv2.imread(v, cv2.IMREAD_GRAYSCALE), (16, 32))
            axes1[idx // 5, idx % 5].imshow(np.concatenate([img_v, np.roll(img_iii, -50, axis=1)], axis=1), cmap='gray')
        plt.show()
    pass

if __name__ == "__main__":
    #exp_iv()
    # exp_ii()
    # exp_iii()
    # exp_iv()
    # exp_vii()
    exp_viii()
