import os
from glob import glob
import shutil

parent_dir = "D:/datacentric_competition_v3/submission_2708_v2/train"
print(os.listdir(parent_dir))

aug_list = glob(os.path.join(parent_dir, '*', '*f_*.png'))
print(len(aug_list))
for item in aug_list:
    os.remove(item)