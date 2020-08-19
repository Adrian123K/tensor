import os
import cv2
import csv
import numpy as np

def image_load(path):
    file_list = os.listdir(path)
    file_list = np.array(list(map(lambda x: cv2.imread(path + '/' + x), file_list)))
    return file_list

def label_load(path):
    file = open(path)
    labeldata = csv.reader(file)
    label = np.array(list(labeldata))
    label = label.astype(int)
    label = np.eye(2)[label]
    return label.reshape(-1, 2)

def next_batch(data1, data2, init, final):
    return data1[init:final], data2[init:final]

def shuffle_batch(data_list, label):
    x = np.arange(len(data_list))
    np.random.shuffle(x)
    data_list2 = data_list[x]
    label2 = label[x]
    return data_list2, label2