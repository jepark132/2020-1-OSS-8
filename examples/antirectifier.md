
## 이 예제는 케라스에서 어떻게 사용자 정의 층을 작성하는지를 보여줍니다.

우리는 'Antirectifier'라는 사용자 정의 활성화층을 빌드합니다.
'Antirectifier'는 그것을 지나가는 텐서의 모양을 수정합니다.
우리는 `compute_output_shape` 와 `call` 두 메소드를 구체화 해야합니다.

람다 층에서도 같은 결과를 얻을 수 있다는 점을 알아두시기 바랍니다.

우리의 사용자 정의 층은 케라스 백엔드의 (`K`)로부터 가져온 
기초요소로 작성되었기 때문에 코드는 텐서플로우와 테아노 두 곳에서
실행될 수 있습니다.

```python
from __future__ import print_function
import keras
from keras.models import Sequential
from keras import layers
from keras.datasets import mnist
from keras import backend as K


class Antirectifier(layers.Layer):
    '''
    이것은 sample-wise L2 정규화와 입력의 양의 부분과 음의 부분의 연속을 조합한 것입니다.
    결과는 입력 샘플보다 두 배 큰 샘플의 텐서입니다.

    이것은 ReLU 대신에 사용될 수 있습니다.

    # 입력 형태
        2D 텐서 형태 (samples, n)

    # 출력 형태
        2D 텐서 형태 (samples, 2*n)

    # 이론적 증명
        ReLU를 적용할 때 이전 출력의 분포가 대략 0의
        중심에 있다고 가정하면 입력의 절반을 폐기하는 것입니다.
        이것은 비효율적이라 할 수 있습니다.

        Antirectifier 는 ReLU 처럼 전체가 양인 출력을
        데이터 폐기 없이 반환할 수 있습니다.

        MNIST에 대한 테스트에 따르면, Antirectifier는 
        등가 ReLU 기반 네트워크로서 유사한 분류 정확도로
        매개변수가 2배 적은 네트워크를 학습시킬 수 있습니다.
    '''

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  # 2D 텐서에서만 유효합니다.
        shape[-1] *= 2
        return tuple(shape)

    def call(self, inputs):
        inputs -= K.mean(inputs, axis=1, keepdims=True)
        inputs = K.l2_normalize(inputs, axis=1)
        pos = K.relu(inputs)
        neg = K.relu(-inputs)
        return K.concatenate([pos, neg], axis=1)

# 전역 파라미터
batch_size = 128
num_classes = 10
epochs = 40

# 데이터는 학습과 테스트셋 사이에서 나뉘어집니다.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 클래스 벡터를 이진 클래스 행렬로 변환합니다.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 모델을 빌드합니다
model = Sequential()
model.add(layers.Dense(256, input_shape=(784,)))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(256))
model.add(Antirectifier())
model.add(layers.Dropout(0.1))
model.add(layers.Dense(num_classes))
model.add(layers.Activation('softmax'))

# 모델을 컴파일합니다
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# 모델을 학습시킵니다
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# 다음은, 두 배 더 큰 Dense층과 ReLU를 가진 동등한 네트워크와 비교합니다

```
