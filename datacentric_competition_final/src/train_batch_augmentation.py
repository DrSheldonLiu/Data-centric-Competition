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


class BatchAugmentation:
    """ do batch augmentation on data-centric dataset """
    digits_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    tar_row, tar_col = 600, 600

    def __init__(self, parent_dir, tar_ext="_generated_dataset",
                 test_path="../label_book", random_name=True, add_weak_learning=True):
        self.tar_ext = tar_ext
        self.test_path = test_path
        self.parent_dir = parent_dir
        self.add_weak_learning = add_weak_learning
        self.src_images, self.src_images_path_dict, self.wl_png_dict = self._set_up()
        self.hard_case_dir = "D:/tars_analysis_both_failed_100_percent"
        self.hard_case_path_dict = self.get_hard_case_path_dict()
        self.name_ext = "-ce5d-11eb-b317-38f9d35ea60f"
        self.random_name = random_name
        # self.image_control()

    def get_hard_case_path_dict(self):
        return {i: glob(os.path.join(self.hard_case_dir, i, "*.png")) for i in BatchAugmentation.digits_list}

    @staticmethod
    def get_random_name():
        return random.choices([i for i in string.ascii_lowercase], k=1)[0] + \
               ''.join(random.choices(''.join(list(map(str, list(range(10))))) + string.ascii_lowercase, k=7))

    def image_control(self):
        target_path = self.parent_dir + self.tar_ext
        cur_train_files = glob(os.path.join(target_path, 'train', '*', '*.png'))
        cur_val_files = glob(os.path.join(target_path, 'val', '*', '*.png'))
        pending = int(len(cur_train_files) - np.floor((10000 - len(cur_val_files)) / 32) * 32)
        if pending <= 0:
            pass
        else:
            ave_remove = [int(np.floor(pending / 10))] * 10
            # mod_num = pending -  ave_remove * 10

    def add_test_images(self):
        test_images = glob(os.path.join(self.test_path, '*', '*.png'))
        for item in test_images:
            tar_images = os.path.normpath(item.replace(self.test_path, os.path.join(self.parent_dir + self.tar_ext, 'train')))
            fname = os.path.basename(item)
            new_fname = fname.split('.')[0] + "_test.png"
            new_fname_path = os.path.join(os.path.dirname(tar_images), new_fname)
            shutil.copy(item, new_fname_path)

    def _set_up(self):
        train_path = os.path.join(self.parent_dir, 'train')
        src_images = glob(os.path.join(train_path, '*', '*f.png'))
        src_images_path_dict = {}
        weak_learning_pngs1 = pd.read_csv('weak_learning_dataset/learning_logs.tsv', sep='\t')['png_name'].to_list()
        weak_learning_pngs2 = pd.read_csv('weak_learning_dataset/new_start_SWL_0409_Codalab_logs.tsv', sep='\t')['png_name'].to_list()
        weak_learning_pngs = weak_learning_pngs1 + weak_learning_pngs2
        weak_learning_png_dict = {i: [j for j in weak_learning_pngs if os.path.basename(os.path.dirname(j)) == i]
                                  for i in BatchAugmentation.digits_list}
        for i in self.digits_list:
            src_images_path_dict[i] = glob(os.path.join(train_path, i, '*f.png'))
        return src_images, src_images_path_dict, weak_learning_png_dict

    def sort_out_val(self):
        # check current file number in train
        target_path = self.parent_dir + self.tar_ext
        cur_train_files = glob(os.path.join(target_path, 'train', '*', '*.png'))
        train_sample = len(cur_train_files)
        equi_train_sample = int(np.ceil(train_sample/8) * 8)
        print(f"there are {train_sample} training samples, equivalent to {equi_train_sample} samples")
        val_needed = 10000 - equi_train_sample
        print(f"{val_needed} val samples needed")
        val_needed_actual = int(np.floor(val_needed / 8) * 8)
        print(f"{val_needed_actual} validation can be added")

        ave_count = val_needed_actual // 10
        original_count_list = [ave_count] * 10
        mod_value = val_needed_actual - ave_count * 10
        for i in range(mod_value):
            original_count_list[i] += 1

        for idx, (kk, vv) in enumerate(self.src_images_path_dict.items()):
            shuffle(vv)
            vv_selected = vv[:original_count_list[idx]]
            for each_vv in vv_selected:
                tar_val = each_vv.replace(os.path.basename(self.parent_dir), os.path.basename(target_path)).replace('train', 'val')
                if not os.path.exists(os.path.dirname(tar_val)):
                    os.makedirs(os.path.dirname(tar_val))
                shutil.copy(each_vv, tar_val)

    def __call__(self, *args, **kwargs):
        shuffle(self.src_images)
        s_x, s_y = 0.8, 1.4
        lamda = 0.25

        # leave the space to add the hard cases
        for kk, vv in self.src_images_path_dict.items():
            print(f'processing {kk}')
            if kk in ['i']:
                list(map(functools.partial(self.apply_transform_scale, s_x=s_x, s_y=s_y), vv[:int(len(vv)/2)]))
                list(map(self.apply_transform_affine_left, vv[int(len(vv)/2):]))

                wl_pngs = self.wl_png_dict[kk]
                wl_png_count = len(wl_pngs)
                list(map(self.sort_out_weak_learning, wl_pngs))

                shuffle(vv)
                list(map(self.apply_transform_affine_full, vv[wl_png_count:]))

                # rotate 20% of the images
                shuffle(vv)
                # rotate_cc = int(0.2 * len(vv))
                # list(map(self.apply_transform_rotate, vv[:]))
                list(map(functools.partial(self.apply_transform_shear, lamda=lamda), vv[:]))
            else:
                list(map(functools.partial(self.apply_transform_scale, s_x=s_x, s_y=s_y), vv))

                wl_pngs = self.wl_png_dict[kk]
                wl_png_count = len(wl_pngs)
                list(map(self.sort_out_weak_learning, wl_pngs))

                shuffle(vv)
                list(map(self.apply_transform_affine_left, vv[wl_png_count:]))
                shuffle(vv)

                # rotate_cc = int(0.1 * len(vv))
                # list(map(self.apply_transform_rotate, vv[:rotate_cc]))
                list(map(self.apply_transform_affine_full, vv[:]))

                # list(map(self.apply_transform_rotate, vv[:]))
                list(map(functools.partial(self.apply_transform_shear, lamda=lamda), vv[:]))
        #self.add_test_images()
        self.sort_out_val()

    def sort_out_weak_learning(self, wl_png_path):
        print(f"wl -> {wl_png_path}")
        tar_path = self.parent_dir + self.tar_ext
        tar_png_path = wl_png_path.replace(os.path.basename(self.parent_dir), os.path.basename(tar_path))
        shutil.copy(wl_png_path, tar_png_path)

    def sort_out_hard_case(self, hard_case_path):
        tar_path = self.parent_dir + self.tar_ext
        digit = os.path.basename(os.path.dirname(hard_case_path))
        random_name = BatchAugmentation.get_random_name() + self.name_ext + "_hard.png"
        print('processing ', random_name)
        random_name_in_tar_path = os.path.join(tar_path, 'train', digit, random_name)
        if not os.path.exists(os.path.dirname(random_name_in_tar_path)):
            os.makedirs(os.path.dirname(random_name_in_tar_path))
        shutil.copy(hard_case_path, random_name_in_tar_path)

    @staticmethod
    def run_padding(img):
        row, col = np.array(img).shape
        # tar_row, tar_col = 3 * row, 3 * col
        tar_row, tar_col = 600, 600
        pad_row = (int((tar_row - row) / 2), tar_row - row - int((tar_row - row) / 2))
        pad_col = (int((tar_col - col) / 2), tar_col - col - int((tar_col - col) / 2))

        img = 255 - np.array(ImageOps.pad(Image.fromarray(255 - np.array(img)), (tar_row, tar_col)))
        # tar_row, tar_col = row + 2 * row, col + 2 * col
        # img_ = np.ones((tar_row, tar_col)) * 255
        # img_[row: 2*row, col: 2 * col] = np.array(img)
        return img

    @staticmethod
    def get_pre_and_post_trans(img):
        img_ = np.array(img)
        mask = np.argwhere(img_ < 43)
        min_row, min_col = np.min(mask, axis=0)
        max_row, max_col = np.max(mask, axis=0)

        middle_point = ((min_row + max_row) / 2, (min_col + max_col) / 2)
        middle_point_ref = (300, 300)
        trans_point = (middle_point_ref[0] - middle_point[0], middle_point_ref[1] - middle_point[1])
        pre_trans = np.array([[1, 0, -trans_point[0]], [0, 1, -trans_point[1]], [0, 0, 1]])
        post_trans = np.array([[1, 0, trans_point[0]], [0, 1, trans_point[1]], [0, 0, 1]])
        return pre_trans, post_trans

    @staticmethod
    def mat_reflect(h):
        return np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, 1, -h], [0, 0, 1]])

    @staticmethod
    def mat_rotate(theta, w, h):
        return np.array([[1, 0, w / 2], [0, 1, h / 2], [0, 0, 1]]) @ np.array(
            [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]) @ np.array(
            [[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1]])

    def apply_transform_affine_full(self, src_img):
        trans_fname = "A_Full"
        print(f'processing -> {src_img}')
        img = Image.open(src_img)
        img = BatchAugmentation.run_padding(img)
        mask = np.argwhere(img < 43)
        min_row, min_col = np.min(mask, axis=0)
        max_row, max_col = np.max(mask, axis=0)

        start_col_rand = np.random.randint(10) + 10
        start_row_rand = np.random.randint(20) + 20

        random_middle_col_point = int(BatchAugmentation.tar_col) - np.random.randint(20)
        random_middle_row_point = np.random.randint(50) + 10 + start_row_rand # create the compress effect at the top right point

        # rectify the end point when the roi is slim
        thre = 100
        if abs(max_col - min_col) < thre:
            random_middle_col_point = start_col_rand + max_col - min_col - np.random.randint(10)

        # get the max col, max row point, random col is random shift based on the above point
        bottom_col = random_middle_col_point + np.random.randint(20)
        bottom_row = 500 + np.random.randint(50)

        # rectify the value when the roi is relatively short
        thre = 30
        if abs(max_row - min_row) < thre:
            bottom_row = max_row + np.random.randint(50)

        pts1 = np.float32([[min_col, min_row], [max_col, min_row], [max_col, max_row]])
        pts2 = np.float32([[start_col_rand, start_row_rand],
                           [random_middle_col_point, random_middle_row_point],
                           [bottom_col, bottom_row]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(np.array(img), M, (600, 600), borderValue=255)

        img_aug = Image.fromarray(dst, "L")

        img_name = os.path.basename(src_img)
        tar_img_name = BatchAugmentation.get_random_name() + self.name_ext + ".png" \
            if self.random_name else img_name.split('.')[0] + f"_{trans_fname}.png"

        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(os.path.basename(self.parent_dir),
                                             os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        # tar_img_path = os.path.join(
        #     os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    def apply_transform_affine_left(self, src_img):
        trans_fname = "A_Left"
        print(f'processing -> {src_img}')
        img = Image.open(src_img)
        img = BatchAugmentation.run_padding(img)
        mask = np.argwhere(img < 43)
        min_row, min_col = np.min(mask, axis=0)
        max_row, max_col = np.max(mask, axis=0)

        start_col_rand = np.random.randint(10) + 10
        start_row_rand = np.random.randint(20) + 20

        random_middle_col_point = int(BatchAugmentation.tar_col / 2) - np.random.randint(20)
        random_middle_row_point = np.random.randint(50) + 10

        # rectify the end point when the roi is slim
        thre = 100
        if abs(max_col - min_col) < thre:
            random_middle_col_point = start_col_rand + max_col - min_col - np.random.randint(10)

        # get the max col, max row point, random col is random shift based on the above point
        bottom_col = random_middle_col_point + np.random.randint(20)
        bottom_row = 500 + np.random.randint(50)

        # rectify the value when the roi is relatively short
        thre = 30
        if abs(max_row - min_row) < thre:
            bottom_row = max_row + np.random.randint(50)

        pts1 = np.float32([[min_col, min_row], [max_col, min_row], [max_col, max_row]])
        pts2 = np.float32([[start_col_rand, start_row_rand],
                           [random_middle_col_point, random_middle_row_point],
                           [bottom_col, bottom_row]])
        M = cv2.getAffineTransform(pts1, pts2)
        dst = cv2.warpAffine(np.array(img), M, (600, 600), borderValue=255)
        img_aug = Image.fromarray(dst, "L")

        img_name = os.path.basename(src_img)
        tar_img_name = BatchAugmentation.get_random_name() + self.name_ext + ".png" \
            if self.random_name else img_name.split('.')[0] + f"_{trans_fname}.png"
        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(os.path.basename(self.parent_dir),
                                             os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        # tar_img_path = os.path.join(
        #     os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    def apply_transform_scale(self, src_img, s_x=0.75, s_y=1.25):
        trans_fname = f"scale_sx{s_x}_sy{s_y}"
        print(f'processing -> {src_img}')
        img = Image.open(src_img)
        img = BatchAugmentation.run_padding(img)
        pre_trans, post_trans = BatchAugmentation.get_pre_and_post_trans(img)

        mat_trans = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]])

        img_aug = Image.fromarray(255 - ndi.affine_transform(255 - img, mat_trans), 'L')
        img_aug = img_aug.resize((600, 600))
        img_name = os.path.basename(src_img)
        tar_img_name = BatchAugmentation.get_random_name() + self.name_ext + ".png" \
            if self.random_name else img_name.split('.')[0] + f"_{trans_fname}.png"
        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(os.path.basename(self.parent_dir),
                                             os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        # tar_img_path = os.path.join(
        #     os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    def apply_transform_shear(self, src_img, lamda=0.5):
        trans_fname = f"shear_lamda{lamda}"
        print(f'processing -> {src_img}')
        img = Image.open(src_img)
        img = BatchAugmentation.run_padding(img)
        # pre_trans, post_trans = BatchAugmentation.get_pre_and_post_trans(img)

        mat_trans = np.array([[1, lamda, 0], [0, 1, lamda], [0, 0, 1]])
        # img_aug = Image.fromarray(255 - ndi.affine_transform(255 - img, pre_trans @ mat_trans @ post_trans))
        img_aug = Image.fromarray(255 - ndi.affine_transform(255 - img, mat_trans), 'L')
        img_aug = img_aug.resize((600, 600))
        img_name = os.path.basename(src_img)
        tar_img_name = BatchAugmentation.get_random_name() + self.name_ext + ".png" \
            if self.random_name else img_name.split('.')[0] + f"_{trans_fname}.png"
        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(os.path.basename(self.parent_dir),
                                             os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        # tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))

        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    def apply_transform_rotate(self, src_img):
        flag = np.random.randint(2)
        theta = (30 + np.random.randint(15)) / 180 * np.pi
        theta = theta * -1 if flag == 1 else theta

        trans_fname = f"rotate_theta{theta}"
        print(f'processing -> {src_img}')
        img = Image.open(src_img)
        img = BatchAugmentation.run_padding(img)
        # pre_trans, post_trans = BatchAugmentation.get_pre_and_post_trans(img)
        h, w = img.shape
        mat_trans = BatchAugmentation.mat_rotate(theta, w, h)
        # img_aug = Image.fromarray(255 - ndi.affine_transform(255 - img, pre_trans @ mat_trans @ post_trans))
        img_aug = Image.fromarray(255 - ndi.affine_transform(255 - img, mat_trans), 'L')
        img_aug = img_aug.resize((600, 600))
        img_name = os.path.basename(src_img)
        tar_img_name = BatchAugmentation.get_random_name() + self.name_ext + ".png" \
            if self.random_name else img_name.split('.')[0] + f"_{trans_fname}.png"
        tar_img_path = os.path.join(
            os.path.dirname(src_img).replace(os.path.basename(self.parent_dir),
                                             os.path.basename(self.parent_dir) + self.tar_ext), tar_img_name)
        # tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(os.path.dirname(tar_img_path)):
            os.makedirs(os.path.dirname(tar_img_path))

        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def convert(img):
        img = (img // 43) * 43
        img[img > 43] = 255
        return img


if __name__ == "__main__":
    batch_aug = BatchAugmentation(
        parent_dir="D:/datacentric_competition_v3/new_start",
        tar_ext="_WL_EN_0409", add_weak_learning=True)
    batch_aug()