from glob import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


class BatchAugmentation:
    """ do batch augmentation on data-centric dataset """

    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.src_images = self._set_up()

    def _set_up(self):
        train_path = os.path.join(self.parent_dir, 'train')
        src_images = glob(os.path.join(train_path, '*', '*.png'))
        return src_images

    def __call__(self, *args, **kwargs):
        list(map(self.augment_rotate, self.src_images))
        #list(map(self.augment_invert, self.src_images))
        #list(map(self.augment_edge, self.src_images)) # no good result
        # list(map(self.augment_mor, self.src_images))
        pass

    @staticmethod
    def augment_rotate(src_img):
        print(f'prcessing -> {src_img}')
        aug_method = "rotation"
        img_ = cv2.imread(src_img, cv2.IMREAD_GRAYSCALE)
        rotate_degree = np.random.randint(45)
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
        kernel = np.ones((3, 3), np.uint8)
        img_ = cv2.erode(img_, kernel, iterations=1)
        img_aug = cv2.dilate(img_, kernel, iterations=3)
        # img_aug = BatchAugmentation.convert(img_aug)

        img_name = os.path.basename(src_img)
        tar_img_name = img_name.split('.')[0] + f"_{aug_method}.png"
        tar_img_path = os.path.join(os.path.dirname(src_img), tar_img_name)
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