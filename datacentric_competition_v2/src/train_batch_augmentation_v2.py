from glob import glob
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image, ImageOps
from random import shuffle


class BatchAugmentation:
    """ do batch augmentation on data-centric dataset """
    digits_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']

    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.src_images, self.src_images_path_dict = self._set_up()

    def _set_up(self):
        train_path = os.path.join(self.parent_dir, 'train')
        src_images = glob(os.path.join(train_path, '*', '*f.png'))
        src_images_path_dict = {}
        for i in self.digits_list:
            src_images_path_dict[i] = glob(os.path.join(train_path, i, '*f.png'))
        return src_images, src_images_path_dict

    def __call__(self, *args, **kwargs):
        shuffle(self.src_images)
        for kk, vv in self.src_images_path_dict.items():
            # if kk != 'ii':
            #     continue
            if kk in ['v', 'vi', 'vii', 'viii']:
                pass
            else:
                continue
            print(f'processing {kk}')
            vv = vv[:60]
            if kk in ['i']:
                # random do the augmentation - mirror, fliph, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'fliplr', 'rotate']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['ii', 'iii']:
                # random do the augmentation - mirror, fliph, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'fliplr', 'rotate', 'noise']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([4]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['x']:
                # random do the augmentation - mirror, fliph, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['fliph', 'crop', 'fliplr', 'rotate']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3, 4]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['iv']:
                # random do the augmentation - iv enrich vi, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'iv_enrich_vi', 'rotate']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['vi']:
                # random do the augmentation - vi enrich iv, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'vi_enrich_iv', 'rotate']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['v']:
                # random do the augmentation - fliplr, rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'fliplr', 'rotate']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)

            elif kk in ['vii', 'viii']:
                # random do the augmentation - rotate, invert, crop, morphological, edge, noise, add_dot_lines
                method_list = [f"augment_{i}" for i in ['crop', 'rotate', 'noise']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)
                        if mm == 'rotate':
                            self.__getattribute__(mm)(each_vv)

            elif kk in ['ix']:
                # random do the augmentation - rotate, invert, crop, morphological, edge, noise, add_dot_lines, fliph
                method_list = [f"augment_{i}" for i in ['crop', 'rotate', 'noise']]
                for each_vv in vv:
                    selected_method = random.sample(method_list, k=random.choice([3]))
                    for mm in selected_method:
                        self.__getattribute__(mm)(each_vv)
                        if mm == 'rotate':
                            self.__getattribute__(mm)(each_vv)


    @staticmethod
    def augment_add_dot_lines(src_img):
        print(f'processing -> {src_img}')
        aug_method = 'dot_lines'
        img = Image.open(src_img)
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
                line_line_distance = np.random.randint(low=10, high=20)
                img_[(random_row + idx * line_line_distance): (random_row + idx * line_line_distance + thickness), ::3] = 0
        else:
            for idx in range(random_count):
                thickness = np.random.randint(low=3, high=10)
                line_line_distance = np.random.randint(low=10, high=20)
                img_[::3, (random_col + idx * line_line_distance): (random_col + idx * line_line_distance + thickness)] = 0
        img_aug = Image.fromarray(img_, mode="L")
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def augment_noise(src_img):
        print(f'processing -> {src_img}')
        aug_method = 'noise'
        img = Image.open(src_img)
        img_ = np.array(img, dtype=np.double)
        img_ = 255 - img_
        noise = np.random.rand(*(img_.shape)) * 40
        img_aug = img_ + noise
        img_aug[img_aug > 240] = 240
        img_aug = 255 - img_aug
        img_aug = img_aug.astype(np.uint8)

        def rotate_image(image, angle):
            scale = np.random.rand() * 1.2 + 0.5
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        rotate_degree = np.random.randint(10, 50)
        rotate_flag = np.random.randint(2)
        img_aug = rotate_image(img_aug, rotate_degree if rotate_flag == 0 else 360 - rotate_degree)
        img_aug = Image.fromarray(img_aug, mode="L")
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def augment_crop(src_img):
        print(f'processing -> {src_img}')
        aug_method = 'crop'
        img_ = Image.open(src_img)
        img_ = 255 - np.array(img_)
        mask = (img_ != 0)
        x_min, y_min = np.argwhere(mask).min(axis=0)
        x_max, y_max = np.argwhere(mask).max(axis=0)
        img_aug = Image.fromarray(255 - img_[x_min: x_max, y_min: y_max], mode="L")
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def augment_fliplr(src_img):
        print(f'processing -> {src_img}')
        aug_method = 'mirror'
        img_ = Image.open(src_img)
        img_aug = ImageOps.mirror(img_)
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def augment_fliph(src_img):
        print(f'processing -> {src_img}')
        aug_method = 'fliph'
        img_ = Image.open(src_img)
        img_aug = ImageOps.flip(img_)
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            img_aug.save(tar_img_path)

    @staticmethod
    def augment_rotate(src_img):
        print(f'processing -> {src_img}')
        aug_method = "rotation"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        rotate_degree = np.random.randint(5, 45)
        rotate_flag = np.random.randint(2)

        def rotate_image(image, angle):
            scale = np.random.rand() * 1.5 + 0.1
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        img_aug = rotate_image(img_, rotate_degree if rotate_flag == 0 else 360 - rotate_degree)
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)
        else:
            cv2.imwrite(tar_img_path.replace(aug_method, aug_method + "_v2"), img_aug)


    @staticmethod
    def augment_invert(src_img):
        print(f'processing -> {src_img}')
        aug_method = "invert"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        # run the augmentation
        img_aug = 255 - img_
        # img_aug = BatchAugmentation.convert(img_aug)

        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_mor(src_img):
        print(f'processing -> {src_img}')
        flag = np.random.randint(2)
        aug_method = "morphological"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)

        # run the augmentation
        if flag == 1:
            img_aug = 255 - cv2.dilate(255 - img_, kernel=np.ones((3, 3), dtype=np.int8), iterations=2)
        else:
            img_aug = 255 - cv2.erode(255 - img_, kernel=np.ones((3, 3), dtype=np.int8), iterations=2)

        # img_aug = BatchAugmentation.convert(img_aug)

        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_iv_enrich_vi(src_img_iv):
        print(f'processing -> {src_img_iv}')
        aug_method = "iv_enrich_vi"
        img_iv = cv2.imread(src_img_iv, cv2.IMREAD_GRAYSCALE)
        img_aug = img_iv[:, ::-1]

        img_name = os.path.basename(src_img_iv)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(src_img_iv)), 'vi'), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_vi_enrich_iv(src_img_vi):
        print(f'processing -> {src_img_vi}')
        aug_method = "vi_enrich_iv"
        img_vi = cv2.imread(src_img_vi, cv2.IMREAD_GRAYSCALE)
        img_aug = img_vi[:, ::-1]

        img_name = os.path.basename(src_img_vi)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(src_img_vi)), 'iv'), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_x(src_img_x):
        print(f'processing -> {src_img_x}')
        aug_method = "flip"
        img_x = cv2.imread(src_img_x, cv2.IMREAD_GRAYSCALE)
        img_aug = img_x[::-1, :]

        img_name = os.path.basename(src_img_x)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(src_img_x)), 'x'), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def convert(img):
        img = (img // 43) * 43
        img[img > 43] = 255
        return img

    @staticmethod
    def augment_edge(src_img):
        print(f'processing -> {src_img}')
        aug_method = "edge"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)

        # run the augmentation
        img_ = cv2.Canny(img_, 10, 43)
        x = np.random.randint(2)
        img_aug = img_ if x == 0 else 255 - img_
        # img_aug = BatchAugmentation.convert(img_aug)

        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)


if __name__ == "__main__":
    batch_aug = BatchAugmentation(parent_dir="D:/datacentric_competition_v2/data")
    batch_aug()