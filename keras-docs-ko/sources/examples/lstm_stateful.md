
## 상태저장 LSTM 모델 사용법과 상태저장 vs 상태 비저장 LSTM 성능 비교

모델들은 입력은 길이의 일반적으로 분배된 임의 순서형인 `input_len` 이고,
출력은 윈도우 길이인 `tsteps` 입력에서의 이동평균인 입/출력 쌍에서 학습되었습니다.
`input_len` 과 `tsteps`는 "편집 가능한 파라미터" 섹션에 정의되어있습니다.

더 큰 `tsteps` 값은 LSTM이 입/출력 관계를 이해하기 위해 더 많은 메모리를 요구할 것을 의미합니다.
이 메모리 길이는 `lahead` 변수에 의해 통제됩니다(자세한 사항은 아래를 참고하시면 됩니다).

남은 파라미터는 다음과 같습니다:
- `input_len`: 생성된 입력 순서형의 길이
- `lahead`: LSTM이 각 출력 포인트에서 학습된 입력 순서형의 길이
- `batch_size`, `epochs`: `model.fit(...)`함수 안의 파라미터와 같은 파라미터들

`lahead > 1`일 때, 모델의 입력은 윈도우 길이가 `lahead` 인 것을 감안하여 데이터의 "rolling window view"로 전처리 됩니다.
이것은 `window_shape`와 함께 sklearn의 `view_as_windows`와 같습니다. 

`lahead < tsteps`일 때, 상태저장력이 `lahead`가 n-포인트 평균에 맞추기 위해 준 능력 너머까지 볼 수 있게 하기 때문에 오직 상태저장 LSTM만이 수렴합니다.
상태 비저장 LSTM은 이 능력이 없고, 그러므로 n-포인트 평균을 보기에 충분하지 않은 그것의 `lahead` 파라미터에 의해 제한을 받습니다.

`lahead >= tsteps`일 때, 상태저장과 상태 비저장 LSTM 모두 수렴합니다.

```python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM

# ----------------------------------------------------------
# 편집 가능한 파라미터
# 세부 사항은 스크립트 상단의 문서를 참고하시기 바랍니다.
# ----------------------------------------------------------

# 입력 길이
input_len = 1000

# 이동 평균의 윈도우 길이는 LSTM을 학습 시키는 
# 입/출력 쌍에서의 입력에서 출력값을 생성하곤 합니다.
# 예시) tsteps=2 이고 input=[1, 2, 3, 4, 5] 면,
#       output=[1.5, 2.5, 3.5, 4.5] 가 됩니다.
tsteps = 2

# LSTM이 각 출력 포인트에서 학습되는 입력 순서형의 길이
lahead = 1

# 학습 파라미터들은 "model.fit(...)"으로 전달됩니다.
batch_size = 1
epochs = 10

# ------------
# 주 프로그램
# ------------

print("*" * 33)
if lahead >= tsteps:
    print("STATELESS LSTM WILL ALSO CONVERGE")
else:
    print("STATELESS LSTM WILL NOT CONVERGE")
print("*" * 33)

np.random.seed(1986)

print('Generating Data...')


def gen_uniform_amp(amp=1, xn=10000):
    """Generates uniform random data between
    -amp and +amp
    and of length xn

    # Arguments
        amp: maximum/minimum range of uniform data
        xn: length of series
    """
    data_input = np.random.uniform(-1 * amp, +1 * amp, xn)
    data_input = pd.DataFrame(data_input)
    return data_input

# 출력이 입력의 이동 평균이기 때문에, 출력의
# 초반 포인트들은 숫자가 아닐 것이고, LSTM을
# 학습시키기 전에 생성된 데이터에서 떨어질 것입니다.
# 또한, lahead > 1 이면, "rolling window view"
# 후의 전처리 단계도 몇 포인트는 손실될 것입니다.
# 미적인 이유로, 전처리 이후에 input_len을 유지하기 위해서,
# 손실될 값들을 설명하기 위한 몇 포인트들을 더해줍니다.

to_drop = max(tsteps - 1, lahead - 1)
data_input = gen_uniform_amp(amp=0.1, xn=input_len + to_drop)

# 입력의 n-포인트 평균이 되도록 목표를 설정합니다.
expected_output = data_input.rolling(window=tsteps, center=False).mean()

# lahead > 1 일 때, 입력을 "rolling window view" 로 전환할 필요가 있습니다.
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.repeat.html
if lahead > 1:
    data_input = np.repeat(data_input.values, repeats=lahead, axis=1)
    data_input = pd.DataFrame(data_input)
    for i, c in enumerate(data_input.columns):
        data_input[c] = data_input[c].shift(i)

# 숫자가 아닌 것들을 떨어뜨립니다.
expected_output = expected_output[to_drop:]
data_input = data_input[to_drop:]

print('Input shape:', data_input.shape)
print('Output shape:', expected_output.shape)
print('Input head: ')
print(data_input.head())
print('Output head: ')
print(expected_output.head())
print('Input tail: ')
print(data_input.tail())
print('Output tail: ')
print(expected_output.tail())

print('Plotting input and expected output')
plt.plot(data_input[0][:10], '.')
plt.plot(expected_output[0][:10], '-')
plt.legend(['Input', 'Expected output'])
plt.title('Input')
plt.show()


def create_model(stateful):
    model = Sequential()
    model.add(LSTM(20,
              input_shape=(lahead, 1),
              batch_size=batch_size,
              stateful=stateful))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
    return model

print('Creating Stateful Model...')
model_stateful = create_model(stateful=True)


# 학습/테스트 데이터를 분할합니다.
def split_data(x, y, ratio=0.8):
    to_train = int(input_len * ratio)
    # batch_size와 맞도록 비틀어줍니다.
    to_train -= to_train % batch_size

    x_train = x[:to_train]
    y_train = y[:to_train]
    x_test = x[to_train:]
    y_test = y[to_train:]

    # batch_size와 맞도록 비틀어줍니다.
    to_drop = x.shape[0] % batch_size
    if to_drop > 0:
        x_test = x_test[:-1 * to_drop]
        y_test = y_test[:-1 * to_drop]

    # 형태를 변형시킵니다.
    reshape_3 = lambda x: x.values.reshape((x.shape[0], x.shape[1], 1))
    x_train = reshape_3(x_train)
    x_test = reshape_3(x_test)

    reshape_2 = lambda x: x.values.reshape((x.shape[0], 1))
    y_train = reshape_2(y_train)
    y_test = reshape_2(y_test)

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = split_data(data_input, expected_output)
print('x_train.shape: ', x_train.shape)
print('y_train.shape: ', y_train.shape)
print('x_test.shape: ', x_test.shape)
print('y_test.shape: ', y_test.shape)

print('Training')
for i in range(epochs):
    print('Epoch', i + 1, '/', epochs)
    # 참고 : 배치 안의 샘플 i의 마지막 상태는 다음 배치의
    # 샘플 i의 처음 상태로 사용될 것입니다.
    # 따라서 우리는 data_input에 포함된 본래의 시리즈보다
    # 해상도가 낮은 batch_size 시리즈에서 동시적으로 학습시킬 수 있습니다.
    
    # 이 각 시리즈는 한 스탭 만큼 오프셋 되고 data_input[i::batch_size]에 의해 추출될 수 있습니다.
    model_stateful.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=1,
                       verbose=1,
                       validation_data=(x_test, y_test),
                       shuffle=False)
    model_stateful.reset_states()

print('Predicting')
predicted_stateful = model_stateful.predict(x_test, batch_size=batch_size)

print('Creating Stateless Model...')
model_stateless = create_model(stateful=False)

print('Training')
model_stateless.fit(x_train,
                    y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test),
                    shuffle=False)

print('Predicting')
predicted_stateless = model_stateless.predict(x_test, batch_size=batch_size)

# ----------------------------

print('Plotting Results')
plt.subplot(3, 1, 1)
plt.plot(y_test)
plt.title('Expected')
plt.subplot(3, 1, 2)
# 첫 "tsteps-1"는 "이전" 순서가 없어서 그것들을 예측할 수 없기 때문에 드랍시킵니다.
plt.plot((y_test - predicted_stateful).flatten()[tsteps - 1:])
plt.title('Stateful: Expected - Predicted')
plt.subplot(3, 1, 3)
plt.plot((y_test - predicted_stateless).flatten())
plt.title('Stateless: Expected - Predicted')
plt.show()
```
