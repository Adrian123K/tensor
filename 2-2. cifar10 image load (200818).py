import os
import re
import numpy as np
import cv2


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


train_image = 'd:/tensor/cifar10/train100/'
print(image_load(train_image))

