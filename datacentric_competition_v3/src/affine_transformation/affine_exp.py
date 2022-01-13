from scipy import ndimage as ndi
from skimage.io import imread
import cv2

import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps

parent_dir = "D:/datacentric_competition_v3/new_start/train"
digit = 'i'
png_list = os.listdir(os.path.join(parent_dir, digit))

# sample_png = png_list[0]
# sample_png_ = "b181b99a-ce5d-11eb-b317-38f9d35ea60f.png"
for sample_png in png_list:
    sample_png_path = os.path.join(parent_dir, digit, sample_png)

    img = Image.open(sample_png_path)
    fig, axes = plt.subplots(1, 3)

    # mat_identity = np.array([[1,0,0],[0,1,0],[0,0,1]])
    # img = 255 - ndi.affine_transform(255 - np.array(img), mat_identity)
    axes[0].imshow(img, cmap='gray')
    axes[0].title.set_text('original')

    row, col = np.array(img).shape
    print('original image height ', row, ' width is ', col)
    tar_row, tar_col = 2 * row, 2 * col
    pad_row = (int((tar_row - row) / 2), tar_row - row - int((tar_row - row) / 2))
    pad_col = (int((tar_col - col) / 2), tar_col - col - int((tar_col - col) / 2))
    # pad_row = (int(row/2), int(row/2))
    # pad_col = (int(col/2), int(col/2))
    # img = np.pad(img, (pad_row, pad_col), constant_values=255)
    # print(img.shape)
    img = Image.fromarray(255 - np.array(ImageOps.pad(Image.fromarray(255 - np.array(img)), (600, 600))))

    h, w = np.array(img).shape
    print('height is ', h, ' width is ', w)

    img_ = np.array(img)
    mask = np.argwhere(img_ < 43)
    min_row, min_col = np.min(mask, axis=0)
    max_row, max_col = np.max(mask, axis=0)
    print(min_row, min_col)
    print(max_row, max_col)
    # plt.figure(); plt.imshow(img_[min_row: max_row, min_col: max_col], cmap='gray')
    # exit(0)

    middle_point = ((min_row + max_row) / 2, (min_col + max_col) / 2)
    middle_point_ref = (300, 300)
    trans_point = (middle_point_ref[0] - middle_point[0], middle_point_ref[1] - middle_point[1])
    pre_trans = np.array([[1, 0, -trans_point[0]], [0, 1, -trans_point[1]], [0, 0, 1]])
    post_trans = np.array([[1, 0, -trans_point[0]], [0, 1, -trans_point[1]], [0, 0, 1]])
    img_roi_centered = ndi.affine_transform(255 - np.array(img), pre_trans)
    axes[1].imshow(255 - img_roi_centered, cmap='gray')

    pts1 = np.float32([[min_col, min_row], [max_col, min_row], [max_col, max_row]])
    pts2 = np.float32([[10, 10], [600, 10], [500, 500]])
    M = cv2.getAffineTransform(pts1, pts2)
    dst = cv2.warpAffine(np.array(img), M, (600, 600), borderValue=255)
    axes[2].imshow(dst, cmap='gray')
    plt.show()
    continue

    shift_vertical = np.array([[1, 0, 0], [0, 1, -col/3], [0, 0, 1]])
    img1 = ndi.affine_transform(255 - np.array(img), shift_vertical)
    axes[1].imshow(255 - img1, cmap='gray')
    axes[1].title.set_text('shift')

    s_x, s_y = 0.7, 1
    mat_scale = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]])
    img2 = ndi.affine_transform(255 - np.array(img), mat_scale)
    axes[2].imshow(255 - img2, cmap='gray')

    lambda1 = 0.6
    mat_shear = np.array([[1,0,0],[lambda1,1,0],[0,0,1]])
    img5 = ndi.affine_transform(255 - np.array(img), mat_shear)
    axes[3].imshow(255 - img5, cmap='gray')
    plt.show()


    mat_reflect = np.matmul(np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]]),
                            np.array([[1, 0, 0], [0, 1, -h], [0, 0, 1]]))
    img1 = ndi.affine_transform(255 - np.array(img), mat_reflect)
    axes[1].imshow(255 - img1, cmap='gray')


    theta = np.pi/6
    mat_rotate = np.array([[1,0,w/2],[0,1,h/2],[0,0,1]]) @ np.array([[np.cos(theta), -np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]]) @ np.array([[1,0,-w/2],[0,1,-h/2],[0,0,1]])
    img3 = ndi.affine_transform(255 - np.array(img), mat_rotate)
    axes[3].imshow(255 - img3, cmap='gray')

    lambda1 = 0.6
    mat_shear1 = np.array([[1,lambda1,0],[0,1,lambda1],[0,0,1]])
    img4 = ndi.affine_transform(255 - np.array(img), mat_shear1)
    axes[4].imshow(255 - img4, cmap='gray')

    lambda1 = 0.6
    mat_shear2 = np.array([[1,0,0],[lambda1,1,0],[0,0,1]])
    img5 = ndi.affine_transform(255 - np.array(img), mat_shear1)
    axes[5].imshow(255 - img5, cmap='gray')

    # mat_all = mat_identity @ mat_scale @ mat_rotate @ mat_shear1
    # img6 = ndi.affine_transform(255 - np.array(img), mat_all)
    # axes[6].imshow(255 - img6, cmap='gray')

    s_x, s_y = 0.8, 0.9
    mat_scale = np.array([[s_x, 0, 0], [0, s_y, 0], [0, 0, 1]])
    img6 = ndi.affine_transform(255 - np.array(img), mat_scale)
    axes[6].imshow(255 - img6, cmap='gray')

    # w, h = 0, 0
    # theta = np.pi / 6
    # mat_rotate = np.array([[np.cos(theta), np.sin(theta), 0], [np.sin(theta), -np.cos(theta), 0], [0, 0, 1]])
    # mat_shift = np.array([[1, 0, w/2], [0, 1, h/2], [0, 0, 1]])
    # img1 = ndi.affine_transform(255 - np.array(img), mat_rotate)
    #
    # #axes[1].imshow(255 - img1, cmap='gray')
    # axes[1].imshow(255 - img2, cmap='gray')
    plt.show()
