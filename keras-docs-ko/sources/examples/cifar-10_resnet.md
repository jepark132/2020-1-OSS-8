# Trains a ResNet on the CIFAR10 dataset.

ResNet v1: 이미지 인식을위한 Deep Resident 학습
ResNet v2: Deep Residual Networks 의 id mapping


| 모델 |n|200-epoch accuracy|원문서 accuracy|sec/epoch GTX1080Ti|
| ----- | ----: | ----: | ----: | --------------: |
| ResNet20 v1 | 3 | 92.16 % | 91.25 % | 35 |
| ResNet32 v1 | 5 | 92.46 % | 92.49 % | 50 |
| ResNet44 v1 | 7 | 92.50 % | 92.83 % | 70 |
| ResNet56 v1 | 9 | 92.71 % | 93.03 % | 90 |
| ResNet110 v1 | 18 | 92.65 % | 93.39+-.16 % | 165 |
| ResNet164 v1 | 27 | - % | 94.07 % | - |
| ResNet1001 v1 | N/A | - % | 92.39 % | - |


| 모델 |n|200-epoch accuracy|원문서 accuracy|sec/epoch GTX1080Ti|
| ----- | ----: | ----: | ----: | --------------: |
| ResNet20 v2 | 2 | - % | - % | --- |
| ResNet32 v2 | N/A | NA % | NA % | NA |
| ResNet44 v2 | N/A | NA % | NA % | NA |
| ResNet56 v2 | 6 | 93.01 % | NA % | 100 |
| ResNet110 v2 | 12 | 93.15 % | 93.63 % | 180 |
| ResNet164 v2 | 18 | - % | 94.54 % | - |
| ResNet1001 v2 | 111 | - % | 95.08+-.14 % | - |


```python
from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

# 학습 변수
batch_size = 32  # 원문서는 네트워크를 batch_size=128 사이즈로 학습시킨다
epochs = 200
data_augmentation = True
num_classes = 10

# 픽셀 평균을 빼면 정확도가 향상됩니다
subtract_pixel_mean = True

# 모델 변수
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti
#           |v1(v2)| %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  | 3 (2)| 92.16     | 91.25     | -----     | -----     | 35 (---)
# ResNet32  | 5(NA)| 92.46     | 92.49     | NA        | NA        | 50 ( NA)
# ResNet44  | 7(NA)| 92.50     | 92.83     | NA        | NA        | 70 ( NA)
# ResNet56  | 9 (6)| 92.71     | 93.03     | 93.01     | NA        | 90 (100)
# ResNet110 |18(12)| 92.65     | 93.39+-.16| 93.15     | 93.63     | 165(180)
# ResNet164 |27(18)| -----     | 94.07     | -----     | 94.54     | ---(---)
# ResNet1001| (111)| -----     | 92.39     | -----     | 95.08+-.14| ---(---)
# ---------------------------------------------------------------------------
n = 3

# 모델 버전
# 원본 논문: version = 1 (ResNet v1), 개선된 ResNet: version = 2 (ResNet v2)
version = 1

# 제공된 모델 파라미터 n에서 계산 된 깊이
if version == 1:
    depth = n * 6 + 2
elif version == 2:
    depth = n * 9 + 2

# 모델 이름, 깊이, 버전
model_type = 'ResNet%dv%d' % (depth, version)

# CIFAR10 데이터를 불러온다.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 이미지 크기를 입력.
input_shape = x_train.shape[1:]

# 데이저 정규화.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# 빼기 픽셀 평균이 활성화 된 경우
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# class 벡터를 이진 클래스 행렬로 변환.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


def lr_schedule(epoch):
    """학습 속도 일정

   
    학습 속도는 80, 120, 160, 180 시대 이후에 줄어들 것으로 예상됨.
    훈련 중 콜백의 일부로 모든 시대를 자동으로 호출함.

    # Arguments
        epoch (int): 에포치 숫자

    # Returns
        lr (float32): 학습 비율
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr


def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True):
    """2D 컨볼 루션-배치 정규화-활성화 스택 빌더

    # Arguments
        inputs (tensor): 입력 이미지 또는 이전 레이어의 입력 텐서
        num_filters (int): Conv2D 필터 수
        kernel_size (int): Conv2D 사각 커널 치수
        strides (int): Conv2D 사각형 보폭
        activation (string): 활성화 이름
        batch_normalization (bool): 배치 정규화 포함 여부
        conv_first (bool): 전환 수 활성화 (참) 또는
            bn-활성화 전환 (거짓)

    # Returns
        x (tensor): 다음 레이어에 입력으로 텐서
    """
    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x


def resnet_v1(input_shape, depth, num_classes=10):
    """ResNet 버전 1 모델 빌더 [a]
   
    2 x (3 x 3) Conv2D-BN-ReLU의 스택
    마지막 ReLU는 바로 가기 연결 후.
    각 단계의 시작 부분에서 기능 맵 크기가 절반으로 줄어듬 (다운 샘플링).
    걸음 수 = 2 인 컨볼 루션 레이어에 의한 반면 필터의 수는 두 배. 
    각 단계에서 레이어는 동일한 수의 필터를 가지며 같은 수의 필터임.
    기능 맵 크기:
    stage 0: 32x32, 16
    stage 1: 16x16, 32
    stage 2:  8x8,  64    
    매개 변수 수는 [a]의 표 6과 대략 동일:
    ResNet20 0.27M
    ResNet32 0.46M
    ResNet44 0.66M
    ResNet56 0.85M
    ResNet110 1.7M

    # Arguments
        input_shape (tensor): 입력 이미지 텐서의 모양
        depth (int): 핵심 컨볼 루션 레이어의 수
        num_classes (int): 클래스의 수 (CIFAR10 has 10)

    # Returns
        model (Model): Keras 모델 인스턴스
    """
    if (depth - 2) % 6 != 0:
        raise ValueError('depth should be 6n+2 (eg 20, 32, 44 in [a])')
    # 모델 정의 시작.
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None)
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # 바뀐 dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # 상단에 분류기를 추가.
    # v1은 마지막 바로 가기 연결 후 BN을 사용하지 않습니다.
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 인스턴스화 모델.
    model = Model(inputs=inputs, outputs=outputs)
    return model


def resnet_v2(input_shape, depth, num_classes=10):
    """ResNet Version 2 Model builder [b]

   
    (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D 또는 병목 현상 레이어
    First shortcut connection per layer is 1 x 1 Conv2D.
    Second and onwards shortcut connection is identity.
    레이어 당 첫 번째 바로 가기 연결은 1 x 1 Conv2D.
    두 번째 이후의 바로 가기 연결은 ID.  
    각 단계의 시작 부분에서 기능 맵 크기가 절반으로 줄어듬 (다운 샘플링).
    필터 맵의 수는 두 배.
    각 단계에서 레이어는 동일한 수의 필터를 가지며 동일한 필터 맵 크기.
    Features maps sizes:
    conv1  : 32x32,  16
    stage 0: 32x32,  64
    stage 1: 16x16, 128
    stage 2:  8x8,  256

    # Arguments
        input_shape (tensor): 입력 이미지 텐서의 모양
        depth (int): 핵심 컨볼 루션 레이어의 수
        num_classes (int): 클래스의 수 (CIFAR10 has 10)

    # Returns
        model (Model): Keras 
    """
    if (depth - 2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
    # 모델 정의를 시작.
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    # v2는 2 개의 경로로 분할하기 전에 입력에서 BN-ReLU를 사용하여 Conv2D를 수행합니다.
    x = resnet_layer(inputs=inputs,
                     num_filters=num_filters_in,
                     conv_first=True)

    # 잔여 단위의 스택을 인스턴스화
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                if res_block == 0:  # first layer and first stage
                    activation = None
                    batch_normalization = False
            else:
                num_filters_out = num_filters_in * 2
                if res_block == 0:  # first layer but not first stage
                    strides = 2    # downsample

            # 병목 현상 잔여 단위
            y = resnet_layer(inputs=x,
                             num_filters=num_filters_in,
                             kernel_size=1,
                             strides=strides,
                             activation=activation,
                             batch_normalization=batch_normalization,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_in,
                             conv_first=False)
            y = resnet_layer(inputs=y,
                             num_filters=num_filters_out,
                             kernel_size=1,
                             conv_first=False)
            if res_block == 0:
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters_out,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False)
            x = keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # 상단에 분류기를 추가.
    # 풀링하기 전에 v2에 BN-ReLU가 있음
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # 모델 인스턴스화.
    model = Model(inputs=inputs, outputs=outputs)
    return model


if version == 2:
    model = resnet_v2(input_shape=input_shape, depth=depth)
else:
    model = resnet_v1(input_shape=input_shape, depth=depth)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(learning_rate=lr_schedule(0)),
              metrics=['accuracy'])
model.summary()
print(model_type)

# 모델 모델 저장 디렉토리를 준비.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'cifar10_%s_model.{epoch:03d}.h5' % model_type
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# 모델 저장 및 학습 속도 조정을위한 콜백을 준비.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]

# 데이터 확대 여부에 관계없이 학습을 실행.
if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              callbacks=callbacks)
else:
    print('Using real-time data augmentation.')
    # 전처리 및 실시간 데이터 확대를 수행:
    datagen = ImageDataGenerator(
        # 데이터 세트에서 입력 평균을 0으로 설정
        featurewise_center=False,
        # 각 표본 평균을 0으로 설정
        samplewise_center=False,
        # 입력을 데이터 세트의 표준으로 나누기
        featurewise_std_normalization=False,
        # 각 입력을 표준으로 나누기
        samplewise_std_normalization=False,
        # ZCA whitening 지원
        zca_whitening=False,
        # epsilon for ZCA whitening
        zca_epsilon=1e-06,
        # 범위 내에서 이미지를 무작위로 회전합니다 (0-180도)
        rotation_range=0,
        # 이미지를 가로로 임의로 이동
        width_shift_range=0.1,
        # 무작위로 이미지를 수직으로 이동
        height_shift_range=0.1,
        # 랜덤 전단의 설정 범위
        shear_range=0.,
        # 랜덤 줌 범위 설정
        zoom_range=0.,
        # 랜덤 채널 시프트를위한 ​​설정 범위
        channel_shift_range=0.,
        # 입력 경계 밖의 점을 채우기위한 설정 모드
        fill_mode='nearest',
        # value used for fill_mode = "constant"
        cval=0.,
        # 무작위로 이미지 뒤집기
        horizontal_flip=True,
        # 무작위로 이미지 뒤집기
        vertical_flip=False,
        # 스케일링 계수 설정 (다른 변환 전에 적용)
        rescale=None,
        # 각 입력에 적용될 설정 기능
        preprocessing_function=None,
        # "channels_first"또는 "channels_last"이미지 데이터 형식
        data_format=None,
        # 유효성 검사를 위해 예약 된 이미지 비율 (0과 1 사이)
        validation_split=0.0)

    # 기능별 정규화에 필요한 계산량
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # datagen.flow ()로 생성 된 배치에 모델을 맞춥니다.
    model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                        validation_data=(x_test, y_test),
                        epochs=epochs, verbose=1, workers=4,
                        callbacks=callbacks)

# 학습 모델 점수 매기기
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

```
