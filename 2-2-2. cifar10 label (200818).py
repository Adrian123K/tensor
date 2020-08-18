train_label = 'd:/tensor/cifar10/train_label.csv'

import csv
import numpy as np


def label_load(path):
    file = open(path)
    label_data = csv.reader(file)

    label_list = []
    for i in label_data:
        label_list.append(i)

    label = np.eye(10)[np.array(label_list).astype(int)].reshape(-1, 10) # reshape(-1, 10) : 2차원으로 축소

    return label


print(label_load(train_label))