import os
import re
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from PIL import Image, ImageOps

digit = 'i'
original_parent_dir = f"D:/datacentric_competition_v2/original_data/train/{digit}"
original_png_list = os.listdir(original_parent_dir)

sample = original_png_list[0]
sample_path = os.path.join(original_parent_dir, sample)
img = Image.open(sample_path)

fig, axes = plt.subplots(1, 2)
axes[0].imshow(img, cmap='gray')

# img_aug = ImageOps.scale(img, 0.5)
img_array = np.array(img)
# img_array_ = np.ones((2 * img_array.shape[0], 2 * img_array.shape[1])) * 255
# random_row_start = np.random.randint(img_array.shape[0])
# random_col_start = np.random.randint(img_array.shape[1])
# img_array_[
#     random_row_start: random_row_start + img_array.shape[0],
#     random_col_start: random_col_start + img_array.shape[1]] = img_array

# add dot lines to image
img_array[::20, ::] = 0
img_array[:, ::20] = 0


axes[1].imshow(img_array, cmap='gray')
plt.show()