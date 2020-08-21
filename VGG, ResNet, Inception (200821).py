# # 설명:
# from tensorflow.keras.applications import *
#
# mobilenet = MobileNet(weights = None, input_shape = None, include_top = True)
# resnet50 = ResNet50(weights = None, input_shape = None, include_top = True)
# xception = Xception(weights = None, input_shape = None, include_top = True)
#
# # 예제:
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation
# from tensorflow.keras.optimizers import Adam
#
#
#
# # imagenet을 학습한 모델을 불러옵니다.
# vgg16 = VGG16(weights = 'imagenet', input_shape = (32, 32, 3), include_top = False)
# vgg16.summary()

# resnet_50 = ResNet50(weights = 'imagenet', input_shape = (224, 224, 3), include_top = False)
# resnet_50.summary()

-------------------------------------------------------------------------------------------------------------------------------
# 1. 데이터 준비하기

from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 평균과 표준편차는 채널별로 구해줍니다.
x_mean = np.mean(x_train, axis = (0, 1, 2))
x_std = np.std(x_train, axis = (0, 1, 2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.3, random_state = 777)
print('data ready~')

# 2. 데이터 증식시키기

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip = True,
                                   zoom_range = 0.2,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   rotation_range = 30,
                                   fill_mode = 'nearest')

val_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow(x_train, y_train,
                                    batch_size = batch_size)

val_generator = val_datagen.flow(x_val, y_val,
                                batch_size = batch_size)

# 3. 사전 학습된 모델 사용하기

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.applications import VGG16

# imagenet을 학습한 모델을 불러옵니다.
vgg16 = VGG16(weights = 'imagenet', input_shape = (32, 32, 3), include_top = False)
vgg16.summary()


# 4. 모델 동결해제하기

# 끝의 4개의 층만 동결을 해제합니다.
for layer in vgg16.layers[:-4]:
    layer.trainable = False

# 5. 모델 구성학습하기

model = Sequential()
# vgg16 모델을 사용합니다.
model.add(vgg16)
# 분류기를 직접 정의합니다.
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation = 'softmax'))

# model.summary() # 모델의 구조를 확인하세요!

model.compile(optimizer = Adam(1e-4),
             loss = 'sparse_categorical_crossentropy',
             metrics = ['acc'])

def get_step(train_len, batch_size):
    if(train_len % batch_size > 0):
        return train_len // batch_size + 1
    else:
        return train_len // batch_size

history = model.fit(train_generator,
                    epochs = 100,
                    steps_per_epoch = get_step(len(x_train), batch_size),
                    validation_data = val_generator,
                    validation_steps = get_step(len(x_val), batch_size))