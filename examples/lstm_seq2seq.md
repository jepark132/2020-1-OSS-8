# Keras에서 Sequence-to-sequence 예제  (문자수준)

이 스크립트는 기본적인 문자 수준의 " sequence-to-sequence " 모델을 구현하는 방법을 보여준다. 
우리는 이것을 문자 대 문자로 짧은 영어 문장을 짧은 프랑스어 문장으로 번역하는 데 적용한다. 
단어 수준 모델이 이 도메인에서 더 일반적이기 때문에 문자 수준 기계 변환을 하는 것은 상당히 이례적이라는 점에 유의하시오.

### 알고리즘 요약

•	우리는 도메인(영어문장)에서 인풋 시퀀스를 시작하고, 또 다른 도메인(프랑스어 문장)에서 corresponding target sequences를 시작한다.

•	인코더 LSTM은 입력 시퀀스를 2개의 상태 벡터로 변환한다(우리는 마지막 LSTM 상태를 유지하고 출력을 삭제함).

•	디코더 LSTM은 대상 시퀀스를 동일한 시퀀스로 변환하지만 향후 하나의 타임스테프로 오프셋되도록 교육되며, 이러한 맥락에서 "교사 강제력"이라고 불리는 교육 과정이다. 인코더의 상태 벡터를 초기 상태로 사용한다. 효과적으로 디코더는 입력 순서에 따라 조건화된  목표물[t+1...], 주어진 표적[...t]을 생성하는 법을 배운다.<br>

•	추론 모드에서 알 수 없는 입력 시퀀스를 디코딩하고자 할 때 우리는:<br>
   
     o	입력 시퀀스를 상태 벡터로 인코딩<br>
     o	크기 1의 타겟 시퀀스에서 시작(시퀀스의 시작 문자만)<br>
     o	디코더에 상태 벡터 및 1-char 목표 시퀀스를 입력하여 다음 문자에 대한 예측 생성<br>
     o	이러한 예측을 사용하여 다음 문자를 샘플링한다(우리는 단순히 argmax를 사용한다).<br>
     o	샘플 문자를 대상 시퀀스에 추가<br>
     o	시퀀스 종료 문자를 생성하거나 문자 한계에 도달할 때까지 반복<br>

### 데이터 다운로드

영어에서 프랑스어로의 문장 쌍 
많은 깔끔한 문장 쌍 데이터 집합 

### 참조
•	 신경망을 이용한Sequence to Sequence <br>
•	Statical Machine 변환용 RNN 인코더-디코더를 이용한 학습 문구 표현 방법  <br>

```python
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # 훈련을 위한 배치 사이즈
epochs = 100  # 훈련할 epochs의 수
latent_dim = 256  # 인코딩 공간의 잠재적 차원성
num_samples = 10000  # 훈련할 예제의 개수
# 디스크 데이터 txt파일의 경로. 
data_path = 'fra-eng/fra.txt'

# 데이터의 벡터화
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # 우리는 “tab”을 “start sequence” 문자로 사용
    # "\n"는 "end sequence" 문자로 지정.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data 가 decoder_input_data보다 한 단계 앞선다. 
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data 가 한 단계 앞설것이며,
            # 시작 문자를 포함하지 않을 것이다.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# 입력 순서를 처리하고 정의한다.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# `encoder_outputs`은 버리고 상태만 유지 
encoder_states = [state_h, state_c]

# `encoder_states`을 초기 상태로 이용하여 디코더를 설정한다. 
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# 전체 출력 시퀀스를 반환하도록 디코더를 설정했지만
# 그리고 역시 내부 상태도 반환하도록.우리는 이용하지 않는다.
# 훈련 모델의 반환 상태, 하지만추론에 사용할 것이다.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# 모델을 정의한다 
# `encoder_input_data` 와 `decoder_input_data`를 `decoder_target_data`로 바꿀
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# 훈련 시작
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# 모델 저장
model.save('s2s.h5')

# 다음 : 추론 모드(샘플링)
# Here's the drill:
# 1) 입력 인코딩 및 초기 디코더 상태 검색
# 2) 초기 상태를 이용하여 디코더의 한 단계를 실행한다.
# 그리고 "start of sequence"토큰을 타겟으로. 
# 출력값은 다음 타겟 토큰이 될 것이다.
# 3) 현재 목표 토큰 및 현재 상태로 반복

# 샘플링 모델을 정의
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# 시퀀스를 다시 디코딩하는 역방향-조회-토큰 인덱스
# 읽을 수 있는 것.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # 입력값을 상태 벡터로  인코딩하시오.
    states_value = encoder_model.predict(input_seq)

    # 길이가 1인 빈 타겟 순서를 발생시키기.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # 타겟 시퀀스의 첫번째 문자를 시작 문자로 채우기 
    target_seq[0, 0, target_token_index['\t']] = 1.

    # 시퀀스 배치에 대한 샘플링 루프
# (간결하게 하기 위해, 여기서는 크기 1의 배치를 가정한다.)
stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # token을 샘플링한다. 
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 탈출 조건 : 최대 길이가 되거나
        # 또는 멈춤 문자를 찾는다
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # target sequence (길이가 1인)을 업데이트.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # 상태 업데이트
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # 한 과정을 수행하라(훈련 세트의 일부)
    # 해독을 위해
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
```
