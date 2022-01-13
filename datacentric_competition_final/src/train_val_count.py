import os


def get_sample_class_weight(train_val_flag ='train'):
    parent_dir = "D:/datacentric_competition_v3/new_start_sheldon_0409"

    train_val_path = os.path.join(parent_dir, train_val_flag)
    print(train_val_flag)
    return [(i, len(os.listdir(os.path.join(train_val_path, i)))) for i in os.listdir(train_val_path)]


def get_auto_class_weight(label_count_list):
    max_count = sorted(label_count_list, key=lambda x: x[1])[-1][-1]
    label_list = [i[0] for i in label_count_list]
    classweight_list = [max_count / i[-1] for i in label_count_list]
    raw_classweight_dict = dict(zip(label_list, classweight_list))
    print(raw_classweight_dict)
    ordered_label_list = ['i', 'ii', 'iii', 'iv', 'v', 'vi', 'vii', 'viii', 'ix', 'x']
    ordered_classweight_dict = {idx: raw_classweight_dict[i] for idx, i in enumerate(ordered_label_list)}
    return ordered_classweight_dict


if __name__ == "__main__":
    label_count_list = get_sample_class_weight('train')
    total_count = sum([i[-1] for i in label_count_list])
    print(f'total train count: {total_count}, count per class')
    print(label_count_list)

    label_count_list = get_sample_class_weight('val')
    total_count = sum([i[-1] for i in label_count_list])
    print(f'total val count: {total_count}, count per class')
    print(label_count_list)

    classweight = get_auto_class_weight(label_count_list)
    print(classweight)