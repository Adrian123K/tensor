import numpy as np
import loader2

def next_batch(data1, data2, init, final):
    return data1[init:final], data2[init:final]

test_image = 'D:/tensor/cifar10/test/'
test_label = 'D:/tensor/cifar10/test_label.csv'

testX = loader2.image_load(test_image)
testY = loader2.label_load(test_label)

print(next_batch(testX, testY, 0 ,100))


def shuffle_batch(data_list, label):
    x = np.arange(len(data_list))
    random.shuffle(x)

    data_list2 = data_list[x]
    label2 = label[x]

    return data_list2, label2