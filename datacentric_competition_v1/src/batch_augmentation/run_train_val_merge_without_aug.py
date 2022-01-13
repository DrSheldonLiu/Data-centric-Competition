import os
from glob import glob
import shutil

parent_dir = "D:/datacentric_competition/data_cleaned_v2"

train_val_merged_parent_dir =  parent_dir + "_train_val_merged"
if not os.path.exists(train_val_merged_parent_dir):
    os.mkdir(train_val_merged_parent_dir)

parent_dir_train_val_splitted = parent_dir + "_train_val_splitted"
if not os.path.exists(parent_dir_train_val_splitted):
    os.mkdir(parent_dir_train_val_splitted)

print(os.listdir(os.path.join(parent_dir, 'train')))

for each_digit in os.listdir(os.path.join(parent_dir, 'train')):
    digit_dir_path_in_train = os.path.join(parent_dir, 'train', each_digit)
    digit_dir_path_in_val = os.path.join(parent_dir, 'val', each_digit)

    digit_dir_path_in_merged = os.path.join(train_val_merged_parent_dir, each_digit)
    if not os.path.exists(digit_dir_path_in_merged):
        os.mkdir(digit_dir_path_in_merged)

    for f in os.listdir(digit_dir_path_in_train):
        shutil.copy(os.path.join(digit_dir_path_in_train, f), os.path.join(digit_dir_path_in_merged, f))
    for f in os.listdir(digit_dir_path_in_val):
        shutil.copy(os.path.join(digit_dir_path_in_val, f), os.path.join(digit_dir_path_in_merged, f))
