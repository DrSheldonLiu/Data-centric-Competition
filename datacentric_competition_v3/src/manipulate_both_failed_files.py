import pandas as pd
import os
from glob import glob
import shutil

parent_dir = "XX"
df = pd.concat([pd.read_csv(os.path.join(parent_dir, 'eval_v2', i), sep='\t') for i in os.listdir(os.path.join(parent_dir, 'eval_v2'))])
print(df)
print(parent_dir)
print(df.shape[0])

parent_dir_to_store = f"D:/tars_analysis_{parent_dir}"
if not os.path.exists(parent_dir_to_store):
    os.mkdir(parent_dir_to_store)


def parse_img_and_move(img_p_pred):
    print('processing -> ', img_p_pred[0])
    img_p, pred = img_p_pred
    a1, fname = os.path.split(img_p)
    a2, digit = os.path.split(a1)
    a3, train_val = os.path.split(a2)
    a4, foldern = os.path.split(a3)
    new_fname = f"{foldern}_{train_val}_GT_{digit}_Pred_{pred}_{fname}"
    new_fpath = os.path.join(parent_dir_to_store, digit, new_fname)
    if not os.path.exists(os.path.dirname(new_fpath)):
        os.makedirs(os.path.dirname(new_fpath))
    shutil.copy(img_p, new_fpath)


list(map(parse_img_and_move, list(zip(df['png_name'].tolist(), df['pred'].to_list()))))
