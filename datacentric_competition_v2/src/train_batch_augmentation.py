from glob import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle


class BatchAugmentation:
    """ do batch augmentation on data-centric dataset """

    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.src_images, self.src_images_i, self.src_images_iv, self.src_images_vi, self.src_images_x = self._set_up()

    def _set_up(self):
        train_path = os.path.join(self.parent_dir, 'train')
        src_images = glob(os.path.join(train_path, '*', '*f.png'))
        src_images_i = glob(os.path.join(train_path, 'i', '*f.png'))
        src_images_iv = glob(os.path.join(train_path, 'iv', '*f.png'))
        src_images_vi = glob(os.path.join(train_path, 'vi', '*f.png'))
        src_images_x = glob(os.path.join(train_path, 'x', '*f.png'))
        return src_images, src_images_i, src_images_iv, src_images_vi, src_images_x

    def __call__(self, *args, **kwargs):
        shuffle(self.src_images)
        img_count = len(self.src_images)
        list(map(self.augment_invert, self.src_images[: int(4/5 * img_count)]))
        list(map(self.augment_rotate, self.src_images[:]))
        list(map(self.augment_mor, self.src_images))

        shuffle(self.src_images_i)
        shuffle(self.src_images_iv)
        shuffle(self.src_images_vi)
        img_i_count = len(self.src_images_i)
        img_iv_count = len(self.src_images_iv)
        img_vi_count = len(self.src_images_vi)

        list(map(self.augment_i_enrich_ii, self.src_images_i[: int(img_i_count)]))
        list(map(self.augment_i_enrich_iii, self.src_images_i[: int(img_i_count)]))
        list(map(self.augment_iv_enrich_vi, self.src_images_iv))
        list(map(self.augment_vi_enrich_iv, self.src_images_vi))
        list(map(self.augment_x, self.src_images_x))
        # list(map(self.augment_edge, self.src_images[int(img_count * 2 / 3):])) # no good result
        pass

    @staticmethod
    def augment_rotate(src_img):
        print(f'prcessing -> {src_img}')
        aug_method = "rotation"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        rotate_degree = np.random.randint(5, 45)
        rotate_flag = np.random.randint(2)

        def rotate_image(image, angle):
            image_center = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
            return result

        img_aug = rotate_image(img_, rotate_degree if rotate_flag == 0 else 360 - rotate_degree)
        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_invert(src_img):
        print(f'prcessing -> {src_img}')
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
        print(f'prcessing -> {src_img}')
        aug_method = "morphological"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)

        # run the augmentation
        img_aug = 255 - cv2.dilate(255 - img_, kernel=np.ones((3, 3), dtype=np.int8), iterations=3)
        # img_aug = BatchAugmentation.convert(img_aug)

        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_i_enrich_ii(src_img_i):
        print(f'processing -> {src_img_i}')
        aug_method = "i_enrich_ii"
        img_one = cv2.imread(src_img_i, cv2.IMREAD_GRAYSCALE)
        img_aug = np.zeros(img_one.shape)
        img_one_rolled = np.roll(img_one, 30, axis=1)
        img_aug[img_one < img_one_rolled] = img_one[img_one < img_one_rolled]
        img_aug[img_one >= img_one_rolled] = img_one_rolled[img_one >= img_one_rolled]
        img_name = os.path.basename(src_img_i)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(src_img_i)), 'ii'), tar_img_name)
        if not os.path.exists(tar_img_path):
            cv2.imwrite(tar_img_path, img_aug)

    @staticmethod
    def augment_i_enrich_iii(src_img_i):
        print(f'processing -> {src_img_i}')
        aug_method = "i_enrich_iii"
        img_one = cv2.imread(src_img_i, cv2.IMREAD_GRAYSCALE)
        img_aug = np.zeros(img_one.shape)
        img_one_rolled = np.roll(img_one, 30, axis=1)
        img_aug[img_one < img_one_rolled] = img_one[img_one < img_one_rolled]
        img_aug[img_one >= img_one_rolled] = img_one_rolled[img_one >= img_one_rolled]

        img_one_rolled_v2 = np.roll(img_one, -30, axis=1)
        img_aug_v2 = np.zeros(img_one.shape)
        img_aug_v2[img_one_rolled_v2 < img_aug] = img_one_rolled_v2[img_one_rolled_v2 < img_aug]
        img_aug_v2[img_one_rolled_v2 >= img_aug] = img_aug[img_one_rolled_v2 >= img_aug]
        img_aug = img_aug_v2

        img_name = os.path.basename(src_img_i)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(src_img_i)), 'iii'), tar_img_name)
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
        print(f'prcessing -> {src_img}')
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