# 이 스크립트는 합성곱 LSTM 네트워크 사용을 보여줍니다.

이 네트워크는 움직이는 윈도우<sub>moving square</sub>를 포함하는 인위적으로 생성 된 영화의 다음 프레임을 예측하는 데 사용됩니다.

```python
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
import numpy as np
import pylab as plt

# 입력을 영화의 형태(n_frames, width, height, channels)로 가지고
# 동일한 모양의 동영상을 반환하는 레이어를 만듭니다.

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   input_shape=(None, 40, 40, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(3, 3),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(Conv3D(filters=1, kernel_size=(3, 3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='binary_crossentropy', optimizer='adadelta')


# 인공 데이터 생성:
# 내부에 3~7 개의 윈도우가 있는 동영상 생성.
# 움직이는 윈도우는 1x1 또는 2x2 픽셀 모양으로 시간이
# 지남에 따라 선형으로 이동합니다.
# 편의상 먼저 너비와 높이가 더 큰 영화 (80x80)를 만들고
# 마지막에 40x40 윈도우를 선택합니다.

def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # 3~7개의 움직이는 윈도우 생성.
        n = np.random.randint(3, 8)

        for j in range(n):
            # 시작 위치
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # 움직이는 방향
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # 윈도우 크기.
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # 노이즈를 추가하여 이상치로 영향을 줄입니다.
                # 아이디어는 인터페이스 동안, 픽셀의 값이 정확히
                # 하나가 아니라면, 네트워크를 견고하게 훈련시키고
                # 윈도우에 속하는 픽셀로 간주해야 한다는 것 입니다.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # ground truth를 1만큼 이동.
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1

    # 40x40의 윈도우로 자르기
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

# 네트워크 학습
noisy_movies, shifted_movies = generate_movies(n_samples=1200)
seq.fit(noisy_movies[:1000], shifted_movies[:1000], batch_size=10,
        epochs=300, validation_split=0.05)

# 하나의 영화에서 네트워크를 테스트하면
# 처음 7 개의 위치로 네트워크를 공급 한
# 다음 새 위치를 예측합니다.
which = 1004
track = noisy_movies[which][:7, ::, ::, ::]

for j in range(16):
    new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
    new = new_pos[::, -1, ::, ::, ::]
    track = np.concatenate((track, new), axis=0)


# 그런 다음 예측을 ground truth와
# 비교하십시오.
track2 = noisy_movies[which][::, ::, ::, ::]
for i in range(15):
    fig = plt.figure(figsize=(10, 5))

    ax = fig.add_subplot(121)

    if i >= 7:
        ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
    else:
        ax.text(1, 3, 'Initial trajectory', fontsize=20)

    toplot = track[i, ::, ::, 0]

    plt.imshow(toplot)
    ax = fig.add_subplot(122)
    plt.text(1, 3, 'Ground truth', fontsize=20)

    toplot = track2[i, ::, ::, 0]
    if i >= 2:
        toplot = shifted_movies[which][i - 1, ::, ::, 0]

    plt.imshow(toplot)
    plt.savefig('%i_animate.png' % (i + 1))

```