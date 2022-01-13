import pandas as pd
import os
from glob import glob
import shutil

parent_dir = "iter3_002"
df_iter = pd.concat([pd.read_csv(os.path.join(parent_dir, 'eval', i), sep='\t') for i in os.listdir(os.path.join(parent_dir, 'eval'))])

parent_dir = "xx_submission_3008_v2"
df_xx = pd.concat([pd.read_csv(os.path.join(parent_dir, 'eval', i), sep='\t') for i in os.listdir(os.path.join(parent_dir, 'eval'))])

df = pd.merge(df_xx, df_iter[['png_name', 'pred']], on=['png_name'], how='inner', suffixes=('_xx', '_iter'))

parent_dir_to_store = "D:/tars_analysis_both_failed"
if not os.path.exists(parent_dir_to_store):
    os.mkdir(parent_dir_to_store)


def parse_img_and_move(img_p_pred):
    print('processing -> ', img_p_pred[0])
    img_p, pred_xx, pred_iter = img_p_pred
    a1, fname = os.path.split(img_p)
    a2, digit = os.path.split(a1)
    a3, train_val = os.path.split(a2)
    a4, foldern = os.path.split(a3)
    new_fname = f"{foldern}_{train_val}_GT_{digit}_Predxx_{pred_xx}_PredIter_{pred_iter}_{fname}"
    new_fpath = os.path.join(parent_dir_to_store, digit, new_fname)
    if not os.path.exists(os.path.dirname(new_fpath)):
        os.makedirs(os.path.dirname(new_fpath))
    shutil.copy(img_p, new_fpath)


list(map(parse_img_and_move, list(zip(df['png_name'].tolist(), df['pred_xx'].to_list(), df['pred_iter'].to_list()))))
