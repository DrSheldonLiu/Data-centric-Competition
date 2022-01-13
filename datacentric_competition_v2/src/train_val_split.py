import os
from random import shuffle
import shutil


def dir_manager(dd):
    if not os.path.exists(dd):
        os.mkdir(dd)


parent_dir = "D:/datacentric_competition_v2"
merged_dir = os.path.join(parent_dir, 'data_merged')
train_val_splitted_dir = os.path.join(parent_dir, 'data')

if not os.path.exists(train_val_splitted_dir):
    os.mkdir(train_val_splitted_dir)

train_in_splitted_dir = os.path.join(train_val_splitted_dir, 'train')
val_in_splitted_dir = os.path.join(train_val_splitted_dir, 'val')
dir_manager(train_in_splitted_dir)
dir_manager(val_in_splitted_dir)

val_ratio = 0.2

for digit in os.listdir(merged_dir):
    digit_in_merged_dir = os.path.join(merged_dir, digit)
    digit_in_train_splitted_dir = os.path.join(train_in_splitted_dir, digit)
    digit_in_val_splitted_dir = os.path.join(val_in_splitted_dir, digit)
    dir_manager(digit_in_train_splitted_dir)
    dir_manager(digit_in_val_splitted_dir)

    src_imgs = os.listdir(digit_in_merged_dir)
    count = len(src_imgs)
    shuffle(src_imgs)
    train_c = int(count * (1-val_ratio))
    train_imgs, val_imgs = src_imgs[:train_c], src_imgs[train_c:]
    for i in train_imgs:
        shutil.move(os.path.join(digit_in_merged_dir, i), os.path.join(digit_in_train_splitted_dir, i))
    for i in val_imgs:
        shutil.move(os.path.join(digit_in_merged_dir, i), os.path.join(digit_in_val_splitted_dir, i))