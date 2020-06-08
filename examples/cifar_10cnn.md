# Train a simple deep CNN on the CIFAR10 small images dataset.


25 에포크에서 75%의 검증 정확도, 50 에포크에서 79%에 달합니다.(그러나 여전히 이 시점에서는 정확하지 않습니다).

```python
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'keras_cifar10_trained_model.h5'

# 학습과 테스트 세트로 분리 된 데이터:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# class 벡터를 binary class 행렬로 변환
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# RMSprop optimizer 초기화
opt = keras.optimizers.RMSprop(learning_rate=0.0001, decay=1e-6)

# RMSprop 모델을 사용하여 학습
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # 전처리 및 실시간 데이터 확대 수행:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # 데이터 세트에서 입력 평균을 0으로 설정
        samplewise_center=False,  # 각각의 샘플 평균 0으로 설정
        featurewise_std_normalization=False,  # 입력을 데이터 세트의 표준으로 나누기
        samplewise_std_normalization=False,  # 각 입력을 표준으로 나누기
        zca_whitening=False,  # ZCA whitening 지원
        zca_epsilon=1e-06,  # ZCA whitening의 엡실론 값
        rotation_range=0,  # 범위 내에서 이미지를 무작위로 회전 (degrees, 0 to 180)
        # 이미지를 가로로 임의로 이동 (fraction of total width)
        width_shift_range=0.1,
        # 이미지를 세로로 임의로 이동 (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # random_shear 범위 설정
        zoom_range=0.,  # zoom_range 범위 설정
        channel_shift_range=0.,  # random_channel_shift 범위 설정
        # 입력 경계 밖의 점을 채우기위한 설정 모드
        fill_mode='nearest',
        cval=0.,  # fill_mode = "constant" 에서 사용될 변수
        horizontal_flip=True,  #  무작위로 이미지 뒤집기
        vertical_flip=False,  # 무작위로 이미지 뒤집기
        # rescaling factor 설정 (다른 변환 전에 적용)
        rescale=None,
        # 각 입력에 적용될 설정 기능
        preprocessing_function=None,
        # 이미지 데이터 형식 "channels_first" 또는 "channels_last"
        data_format=None,
        # 유효성 검사를 위해 예약 된 이미지의 일부 (0과 1 사이)
        validation_split=0.0)

    # 기능별 정규화에 필요한 계산
    # (ZAC whitening 이 적용되었을때 std, mean, principal 구성요소).
    datagen.fit(x_train)

    # datagen.flow ()로 생성 된 배치에 모델을 맞춤
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# 모델과 점수 저장
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# 학습 모델 점수 매김
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
```
