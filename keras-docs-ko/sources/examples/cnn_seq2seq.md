예측을 생성하려면 문자 수준 시퀀스를 시퀀스 모델로 복원하십시오.
이 스크립트는  lstm_seq2seq.py 에 의해 저장된  s2s.h5  모델을 로드하고 그것으로부터 시퀀스를 생성한다. 그것은 어떠한 변경도 이루어지지 않았다고 가정한다(예: latent_dim  은 변경되지 않고, 입력 데이터와 모델 아키텍처는 변경되지 않는다).
모델 아키텍처 및 모델 아키텍처 교육 방법에 대한 자세한 내용은  lstm_seq2seq.py를 참조하십시오.
from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input
import numpy as np

batch_size = 64  # 훈련을 위한 배치 사이즈
epochs = 100  # 훈련할 epochs의 수
latent_dim = 256  # 인코딩 공간의 잠재적 차원성
num_samples = 10000  # 훈련할 예제의 개수
# 디스크 데이터 txt파일의 경로. 
data_path = 'fra-eng/fra.txt'

# 데이터의 벡터화
 우리는 훈련 시나리오와 같은 접근법을 사용한다.
# 참고 : 데이터는 문자 -> 정수 순서로 동일해야 한다.
# 일관성을 유지하기 위한 매핑.
# target_texts는 필요하지 않으므로 인코딩을 생략한다.
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

for i, input_text in enumerate(input_texts):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.

# 모델 복원 및 인코더 및 디코더 구성
model = load_model('s2s.h5')

encoder_inputs = model.input[0]   # 1을 인풋
encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output   # lstm_1
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # 2를 인풋
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[3]
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[4]
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


# 인풋 시퀀스를 디코딩하라. 후의 작업들은 beam 탐색을 지지해줄 것.
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

        # 토큰을 샘플링
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # 탈출 조건 : 최대 길이에 도달하거나
        # 멈춤 문자를 찾을 것
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # 타겟 시퀀스(길이가 1인)를 업데이트
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

