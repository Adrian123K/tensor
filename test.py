import loader2

train_image='D:/tensor/cifar10/train/'
train_label = 'D:/tensor/cifar10/train_label.csv'
test_image='D:/tensor/cifar10/test/'
test_label = 'D:/tensor/cifar10/test_label.csv'


trainX = loader2.image_load(train_image)
trainY = loader2.label_load(train_label)
testX = loader2.image_load(test_image)
testY = loader2.label_load(test_label)

testX, testY = loader2.shuffle_batch(testX, testY)
print(loader2.next_batch(testX, testY, 0, 100))