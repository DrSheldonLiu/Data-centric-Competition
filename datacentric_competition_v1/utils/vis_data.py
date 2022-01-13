import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import pandas as pd
from glob import glob
from random import shuffle

from pylab import rcParams


class vis_data:
    def __init__(self, parent_dir):
        self.parent_dir = parent_dir
        self.qc_logdir: str = self.qc_logdir_manager()

    def __call__(self, flag='single', group_size=25, col=None, *args, **kwargs):
        if flag == 'single':
            # run img vis per digit
            digit_dict = self._get_per_number_img_list()
            row = int(np.sqrt(group_size)) if col is None else group_size // col
            col = group_size // row if col is None else col
            for k, v in digit_dict.items():
                self.qc_list = []
                for each_split in [v[i: i + group_size] for i in range(0, len(v), group_size)]:
                    fig, axes = plt.subplots(row, col, figsize=(20, 10))
                    counter = 0
                    for img_path in each_split:
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        axes[counter // col, counter % col].imshow(img, cmap='gray')
                        axes[counter // col, counter % col].img_name = img_path
                        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
                        counter += 1
                    plt.suptitle(k, fontsize=18)
                    plt.show()
                qc_fname = f"{os.path.basename(self.parent_dir)}_{k}.tsv"
                pd.DataFrame(self.qc_list, columns=['QC_List']).to_csv(
                    os.path.join(self.qc_logdir, qc_fname), sep='\t', index=False)


                    #plt.show(block=False)
                    #plt.pause(4)
                    #plt.close()

        elif flag == 'both':
            digit_dict = self._get_train_val_comp_pair()
            row = int(np.sqrt(group_size)) if col is None else group_size // col
            col = group_size // row if col is None else col
            for k, v in digit_dict.items():
                counter = 0

                fig_width = 10
                fig_height = 10

                shuffle(v['train'])
                shuffle(v['val'])
                fig_train, axes_train = plt.subplots(5, 5, figsize=(fig_width, fig_height))
                plt.suptitle('train', fontsize=15)
                fig_val, axes_val = plt.subplots(5, 5, figsize=(fig_width, fig_height))
                plt.suptitle('val', fontsize=15)

                for train_img_path, val_img_path in zip(v['train'][:25], v['val'][:25]):
                    img = cv2.imread(train_img_path, cv2.IMREAD_GRAYSCALE)
                    axes_train[counter // 5, counter % 5].imshow(img, cmap='gray')

                    img = cv2.imread(val_img_path, cv2.IMREAD_GRAYSCALE)
                    axes_val[counter // 5, counter % 5].imshow(img, cmap='gray')
                    counter += 1
                plt.show()

        else:
            # run vis randomly, either for train or val
            pass

    def qc_logdir_manager(self) -> str:
        log_dir = "qc_logs" if not 'bin' in self.parent_dir else "qc_bin_logs"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return log_dir

    def onclick(self, event):
        #print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
        #      ('double' if event.dblclick else 'single', event.button,
        #       event.x, event.y, event.xdata, event.ydata))
        if not hasattr(event.inaxes, "img_name"):
            return 0

        if not event.dblclick:
            print(event.inaxes.img_name)
            event.inaxes.visible = False
            if os.path.basename(event.inaxes.img_name) not in self.qc_list:
                self.qc_list.append(os.path.basename(event.inaxes.img_name))
        else:
            event.inaxes.visible = True
            #print('remove image in list, if it exists')
            if os.path.basename(event.inaxes.img_name) in self.qc_list:
                print(f'removing --- {os.path.basename(event.inaxes.img_name)}')
                self.qc_list.remove(os.path.basename(event.inaxes.img_name))
        print('current qc list: ', self.qc_list)

    @staticmethod
    def decode_path(img_path):
        return os.path.basename(os.path.dirname(img_path))

    def _get_png_pair(self):
        raw_img_list = glob(os.path.join(self.parent_dir, "*", "*.png"))
        raw_img_pair = [(vis_data.decode_path(i), i) for i in raw_img_list]
        return raw_img_pair

    def _get_per_number_img_list(self):
        raw_digit_list = [i for i in os.listdir(self.parent_dir)]
        raw_digit_dict = {i: glob(os.path.join(self.parent_dir, i, '*.png')) for i in raw_digit_list}
        return raw_digit_dict

    def _get_train_val_comp_pair(self):
        # in this mode, parent dir is till the one above both train and val
        raw_digit_list = [i for i in os.listdir(os.path.join(self.parent_dir, 'train'))]
        raw_digit_dict = {
            i: {'train': glob(os.path.join(self.parent_dir, 'train', i, '*.png')),
                'val': glob(os.path.join(self.parent_dir, 'val', i, '*.png'))}
            for i in raw_digit_list if not i.startswith('.')}
        return raw_digit_dict


if __name__ == "__main__":
    # ------------------------------------ #
    # vis training data, per digit
    vis = vis_data(parent_dir="D:/datacentric_competition/data_cleaned/train_bin")
    vis(flag='single', group_size=100, col=10)

    # ------------------------------------ #
    # vis val data, per digit
    # vis = vis_data(parent_dir="D:/datacentric_competition/data_cleaned/val")
    # vis(flag='single', group_size=60, col=10)

    # ------------------------------------ #
    # vis train vs val
    #vis = vis_data(parent_dir='D:/datacentric_competition/data_cleaned')
    #vis(flag='both')