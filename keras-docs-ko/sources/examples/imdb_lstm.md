```python
## IMDB 감정 분류 작업에서의 LSTM 모델 학습

데이터셋이 LSTM에 대해 너무 작아서 TF-IDF + LogReg 와 같이 간단하고
훨씬 빠른 방법에 비해 어떠한 이점도 가질 수 없습니다.

### 참고

- RNN은 까다롭습니다. 중요한 것은 배치 사이즈를 결정하는 것과, 손실과 최적화 함수의 결정 등이 있습니다. 어떤 배열들은 수렴되지 않습니다. 

- 학습 시 LSTM 손실 감소 패턴은 CNN, MLP 등에서 보이는 양상과 다소 다를 수 있습니다.

```
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.datasets import imdb

max_features = 20000
# 이 문자열 후의 텍스트를 자릅니다 (max_features에서 가장 비슷한 단어들 중에서)
maxlen = 80
batch_size = 32

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
model = Sequential()
model.add(Embedding(max_features, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

# 여러 최적화 함수와 배치들을 사용하십시오
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_data=(x_test, y_test))
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
