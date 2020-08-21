from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np

# # 이미지 제네레이터를 선언합니다.
# train_datagen = ImageDataGenerator(horizontal_flip=True,
#                                    vertical_flip=True,
#                                    shear_range=0.5,
#                                    brightness_range=[0.5, 1.5],
#                                    zoom_range=0.2,
#                                    width_shift_range=0.1,
#                                    height_shift_range=0.1,
#                                    rotation_range=30,
#                                    fill_mode='nearest'
#                                    )
#
# # 햄버거 사진을 불러옵니다.
# hamburger = img_to_array(load_img('d:/data/햄버거.png')).astype(np.uint8)
# plt.figure();
# plt.title('original image')
# plt.imshow(hamburger)
#
# # 제네레이터를 사용해서 이미지를 변환합니다.
# hamburger = hamburger.reshape((1,) + hamburger.shape)
# train_generator = train_datagen.flow(hamburger, batch_size=1)
#
# fig = plt.figure(figsize=(5, 5))
# fig.suptitle('augmented image')
#
# for i in range(9):
#     data = next(train_generator)  # 제네레이터에게서 이미지를 받아옵니다.
#     image = data[0]
#     plt.subplot(3, 3, i + 1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.imshow(np.array(image, dtype=np.uint8), cmap='gray')
#
# plt.show()

# 2. 데이터 증식 이용하여 cifar10 학습 시키기

from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 평균과 표준편차는 채널별로 구해줍니다.
x_mean = np.mean(x_train, axis=(0, 1, 2))
x_std = np.std(x_train, axis=(0, 1, 2))

x_train = (x_train - x_mean) / x_std
x_test = (x_test - x_mean) / x_std

# 3. 이미지 제네레이터를 사용하여 모델 학습하기

from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train,
                                                  test_size=0.3, random_state=777)
print('data ready~')

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(horizontal_flip=True,
                                   zoom_range=0.2,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rotation_range=30,
                                   fill_mode='nearest'
                                   )

# 검증 데이터셋에는 변환을 사용하지 않습니다.
val_datagen = ImageDataGenerator()

batch_size = 32

train_generator = train_datagen.flow(x_train, y_train,
                                     batch_size=batch_size)
val_generator = val_datagen.flow(x_val, y_val,
                                 batch_size=batch_size)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Activation, BatchNormalization
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, padding='same', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=32, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))
model.add(Flatten())
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer=Adam(1e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['acc'])


def get_step(train_len, batch_size):
    if (train_len % batch_size > 0):
        return train_len // batch_size + 1
    else:
        return train_len // batch_size


history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=get_step(len(x_train), batch_size),
                    validation_data=val_generator,
                    validation_steps=get_step(len(x_val), batch_size))

# # 4. 학습 과정 시각화 하기
#
# import matplotlib.pyplot as plt
#
# his_dict = history.history
# loss = his_dict['loss']
# val_loss = his_dict['val_loss']
#
# epochs = range(1, len(loss) + 1)
# fig = plt.figure(figsize=(10, 5))
#
# # 훈련 및 검증 손실 그리기
# ax1 = fig.add_subplot(1, 2, 1)
# ax1.plot(epochs, loss, color='blue', label='train_loss')
# ax1.plot(epochs, val_loss, color='orange', label='val_loss')
# ax1.set_title('train and val loss')
# ax1.set_xlabel('epochs')
# ax1.set_ylabel('loss')
# ax1.legend()
#
# acc = his_dict['acc']
# val_acc = his_dict['val_acc']
#
# # 훈련 및 검증 정확도 그리기
# ax2 = fig.add_subplot(1, 2, 2)
# ax2.plot(epochs, acc, color='blue', label='train_acc')
# ax2.plot(epochs, val_acc, color='orange', label='val_acc')
# ax2.set_title('train and val acc')
# ax2.set_xlabel('epochs')
# ax2.set_ylabel('acc')
# ax2.legend()
#
# plt.show()
#
#
