import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from train_val_count import get_auto_class_weight, get_sample_class_weight
from datetime import datetime
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import argparse
import os
import numpy as np
import json
import sys

#directory = "../data"
directory = r"D:\datacentric_competition_v3\submission\submission_3008_v2"
user_data = valid_data = directory
test_data = "../label_book"

### DO NOT MODIFY BELOW THIS LINE, THIS IS THE FIXED MODEL ###
batch_size = 8 * 4 * 4
tf.random.set_seed(123)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False)
    parser.add_argument('--timestamp', default=False)
    args = parser.parse_args(sys.argv[1:])
    config = tf.config.experimental.list_physical_devices('GPU')
    if len(config) != 0:
        tf.config.experimental.set_memory_growth(config[0], True)

    class_weight = get_auto_class_weight(get_sample_class_weight('train'))
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    ])
    train = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/train',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )
    #train = train.map(lambda image, label: (tf.keras.layers.GaussianNoise(1)(image), label))
    #train = train.map(
    #    lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)).cache().map(
    #    lambda image, label: (tf.image.random_flip_left_right(image), label)).map(
    #    lambda image, label: (tf.image.rot90(image), label))
   # train = train.map(
   #     lambda image, label: (tf.image.convert_image_dtype(image, tf.float32), label)).cache().map(
   #     lambda image, label: (tf.image.random_flip_left_right(image), label))
    #   train = train.map(lambda image, label: (tf.image.random_crop(image, (26, 26, 3))))

    valid = tf.keras.preprocessing.image_dataset_from_directory(
        user_data + '/val',
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    total_length = ((train.cardinality() + valid.cardinality()) * batch_size).numpy()
    total_length = 0
    if total_length > 10_000:
        print(f"Dataset size larger than 10,000. Got {total_length} examples")
        sys.exit()

    test = tf.keras.preprocessing.image_dataset_from_directory(
        test_data,
        labels="inferred",
        label_mode="categorical",
        class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
        shuffle=False,
        seed=123,
        batch_size=batch_size,
        image_size=(32, 32),
    )

    base_model = tf.keras.applications.ResNet50(
        input_shape=(32, 32, 3),
        include_top=False,
        weights=None,
    )
    base_model = tf.keras.Model(
        base_model.inputs, outputs=[base_model.get_layer("conv2_block3_out").output]
    )

    inputs = tf.keras.Input(shape=(32, 32, 3))
    # # adding data augmentation
    # inputs = tf.keras.layers.Lambda(lambda xx: data_augmentation(xx))(inputs)
    # inputs = tf.keras.layers.GaussianNoise(1)(inputs)
    x = tf.keras.applications.resnet.preprocess_input(inputs)
    x = base_model(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(10)(x)
    model = tf.keras.Model(inputs, x)

    if not args.eval:
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )
        model.summary()
        loss_0, acc_0 = model.evaluate(valid)
        print(f"loss {loss_0}, acc {acc_0}")

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        ckpt_path = f"logs/ckpt/{timestamp}/checkpoint"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=ckpt_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=True,
        )

        earlystopping_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', restore_best_weights=True,
            min_delta=0.0001, patience=15)
        logdir = "logs/fit/" + timestamp
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

        history = model.fit(
            train,
            validation_data=valid,
            epochs=100,
            # callbacks=[checkpoint, earlystopping_callback, tensorboard_callback],
            callbacks=[checkpoint, tensorboard_callback],
            # class_weight=class_weight
        )

        model.load_weights(ckpt_path)

        loss, acc = model.evaluate(valid)
        print(f"final loss {loss}, final acc {acc}")

        test_loss, test_acc = model.evaluate(test)
        print(f"test loss {test_loss}, test acc {test_acc}")

    else:
        assert args.timestamp, 'timestamp required'
        ckpt_path = os.path.join('logs', 'ckpt', args.timestamp, 'checkpoint')
        assert os.path.exists(ckpt_path), "checkpoint not found"
        model.load_weights(ckpt_path)
        print('weight loaded')

        def get_pd_dict(ds: tf.data.Dataset, vis=False, tt='val', show_detail=True):
            img_list = np.concatenate(list(ds.map(lambda x, y: x).as_numpy_iterator()))
            raw_gt = np.concatenate(list(ds.map(lambda x, y: y).as_numpy_iterator()))
            gt = np.argmax(raw_gt, axis=1)
            class_name = ds.class_names

            ds = ds.prefetch(tf.data.AUTOTUNE)
            raw_pred = model.predict(ds)
            pred = np.argmax(raw_pred, axis=-1)
            cm = confusion_matrix(gt, pred)

            if show_detail:
                mask = (gt != pred)
                d_imgs = img_list[mask]
                d_imgs = [d_imgs[i, ...] for i in range(d_imgs.shape[0])]
                d_pred = pred[mask]
                d_gt = gt[mask]
                n_sqr = 5
                d_pairs = list(zip(d_imgs, d_pred, d_gt))
                for each_group in [d_pairs[i: i + n_sqr ** 2] for i in range(0, len(d_pairs), n_sqr ** 2)]:
                    fig_d, ax_d = plt.subplots(n_sqr, n_sqr)
                    for idx, each_item in enumerate(each_group):
                        d_img_, d_pred_, d_gt_ = each_item
                        d_pred_name = class_name[d_pred_]
                        d_gt_name = class_name[d_gt_]
                        ax_d[idx // n_sqr, idx % n_sqr].imshow(d_img_ / np.max(d_img_), cmap='gray')
                        ax_d[idx // n_sqr, idx % n_sqr].title.set_text(f"Pred -> {d_pred_name} vs GT -> {d_gt_name}")
                    plt.suptitle(tt)
                    plt.show()

            cm = np.round(np.divide(cm, cm.sum(axis=-1, keepdims=True) + 1e-10), 2)
            disp = ConfusionMatrixDisplay(cm, display_labels=class_name)
            disp.plot(include_values=True)
            #plt.savefig(os.path.join(os.path.dirname(ckpt_path), 'val_cm.jpg'))
            zip_r = list(zip(pred, gt))
            pd_list = [sum([i == j == k for i, j in zip_r]) / sum([j == k for i, j in zip_r]) if sum([j == k for i, j in zip_r]) != 0 else 0 for k in range(10)]
            acc = sum([i == j for i, j in zip_r]) / len(zip_r)
            if vis:
                plt.title(f'overall acc: {acc}')
                plt.show()

            pd_dict = dict(zip(class_name, pd_list))
            return pd_dict, acc

        val_pd_dict, val_acc = get_pd_dict(valid, vis=True, tt='val', show_detail=False)

        train_ = tf.keras.preprocessing.image_dataset_from_directory(
            user_data + '/train',
            labels="inferred",
            label_mode="categorical",
            class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
            shuffle=False,
            seed=123,
            batch_size=batch_size,
            image_size=(32, 32),
        )
        test_ = tf.keras.preprocessing.image_dataset_from_directory(
            test_data,
            labels="inferred",
            label_mode="categorical",
            class_names=["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"],
            shuffle=False,
            seed=123,
            batch_size=batch_size,
            image_size=(32, 32),
            )

       # train_pd_dict, train_acc = get_pd_dict(train_, vis=True, tt='train', show_detail=True)
        test_pd_dict, test_acc = get_pd_dict(test_, vis=True, tt='test', show_detail=True)

        print('-------------------------------------')
        print('validation ->')
        print(f'overall acc : {val_acc}')
        print(f'per class acc: {val_pd_dict}')
        print('\n')

        print('-------------------------------------')
        print('test ->')
        print(f'overall acc: {test_acc}')
        print(f'per class acc: {test_pd_dict}')
        print('\n')
