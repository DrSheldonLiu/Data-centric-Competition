import pandas as pd
import glob
import os
import re
import shutil

clean_parent_dir = "D:/datacentric_competition/data_cleaned"
log_dir = "D:/datacentric_competition/utils/qc_logs"


def parse_path(fpath):
    fname = os.path.basename(fpath)
    matched_ = re.match('(.+)_(.+).tsv', fname)
    train_val_flag, digit_num = matched_.groups()
    tar_dir = train_val_flag + "_bin"
    tar_dir_path = os.path.join(clean_parent_dir, tar_dir)
    if not os.path.exists(tar_dir_path):
        os.mkdir(tar_dir_path)

    digit_num_in_tar = os.path.join(tar_dir_path, digit_num)
    digit_num_in_src = os.path.join(clean_parent_dir, train_val_flag, digit_num)
    if not os.path.exists(digit_num_in_tar):
        os.mkdir(digit_num_in_tar)

    df = pd.read_csv(fpath, sep='\t')

    if df.shape[0] != 0:
        cmd_list = [(f"mv -v {os.path.join(digit_num_in_src, i)} {os.path.join(digit_num_in_tar, i)}",
                     os.path.exists(os.path.join(digit_num_in_src, i))) for i in df['QC_List'].to_list()]
        for item in df['QC_List'].to_list():
            src_path = os.path.join(digit_num_in_src, item)
            tar_path = os.path.join(digit_num_in_tar, item)
            shutil.move(src_path, tar_path)


train_log_tsv_list = glob.glob(os.path.join(log_dir, 'train*.tsv'))
list(map(parse_path, train_log_tsv_list))

val_log_tsv_list = glob.glob(os.path.join(log_dir, 'val*.tsv'))
list(map(parse_path, val_log_tsv_list))

print(train_log_tsv_list)
print(val_log_tsv_list)
