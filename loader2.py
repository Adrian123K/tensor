import os
import re
import cv2
import csv
import numpy as np

def image_load(path):
    file_list = os.listdir(path)

    file_name = []
    for i in file_list:
        file_name.append(int(re.sub('[^0-9]', '', i)))

    file_name.sort()

    file_res = []
    for j in file_name:
        file_res.append(path + str(j) + '.png')

    image = []
    for k in file_res:
        image.append(cv2.imread(k))

    return np.array(image)

def label_load(path):
    file = open(path)
    label_data = csv.reader(file)

    label_list = []
    for i in label_data:
        label_list.append(i)

    label = np.eye(10)[np.array(label_list).astype(int)].reshape(-1, 10) # reshape(-1, 10) : 2차원으로 축소

    return label

def next_batch(data1, data2, init, final):
    return data1[init:final], data2[init:final]


def shuffle_batch(data_list, label):
    x = np.arange(len(data_list))
    np.random.shuffle(x)

    data_list2 = data_list[x]
    label2 = label[x]

    return data_list2, label2