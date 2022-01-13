from glob import glob
import os
import cv2
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

source_dir = "D:/datacentric_competition_v2/data"
train_val_flag = "train"
train_path = os.path.join(source_dir, train_val_flag)
#print(os.listdir(train_path))

target_dir = source_dir + "_augmented_rotate"
if not os.path.exists(target_dir):
    os.mkdir(target_dir)
train_path_in_tar = os.path.join(target_dir, train_val_flag)
if not os.path.exists(train_path_in_tar):
    os.mkdir(train_path_in_tar)

src_images = glob(os.path.join(train_path, '*', "*f.png"))
tar_images = [i.replace(os.path.basename(source_dir), os.path.basename(target_dir)) for i in src_images]


def augment_images(img):
    rotate_degree = np.random.randint(45)
    rotate_flag = np.random.randint(2)

    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
        return result

    img = rotate_image(img, rotate_degree if rotate_flag == 0 else 360 - rotate_degree)
    image = img
    # image = (img // 43) * 43
    # image[image > 43] = 255
    return image


def vis():
    for img_group in [src_images[i: i+25] for i in range(0, len(src_images), 25)]:
        _, ax1 = plt.subplots(5, 5)
        _, ax2 = plt.subplots(5, 5)
        for i in range(25):
            img_ = cv2.imread(img_group[i], cv2.IMREAD_GRAYSCALE)
            ax1[i // 5, i % 5].imshow(img_, cmap='gray')
            ax2[i // 5, i % 5].imshow(augment_images(img_), cmap='gray')
        plt.show()


if __name__ == "__main__":
    vis()
    for i, j in zip(src_images, tar_images):
        print(i, j)
        img_ = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
        img_inverted = augment_images(img_)
        tar_images_dir = os.path.dirname(j)
        if not os.path.exists(tar_images_dir):
            os.mkdir(tar_images_dir)
        tar_images_fname = os.path.basename(j).split('.')[0] + "_rotated" + ".png"
        new_tar_images_path = os.path.join(tar_images_dir, tar_images_fname)
        cv2.imwrite(new_tar_images_path, img_inverted)