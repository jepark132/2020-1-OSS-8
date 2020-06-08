# Trains two recurrent neural networks based upon a story and a question.


병합 된 벡터는 다양한 bAbI 작업에 응답하도록 요구되어진다.

결과는 Weston 등에 의해 제공 되어진 LSTM 모델과 비교 된다.: : "Towards AI-Complete Question Answering: A Set of Prerequisite Toy Tasks" http://arxiv.org/abs/1502.05698


| task 번호 |FB LSTM Baseline|Keras QA|
| ----- | ----: | --------------: |
| QA1 - Single Supporting Fact | 50 | 52.1 |
| QA2 - Two Supporting Facts | 20 | 37.0 |
| QA3 - Three Supporting Facts | 20 | 20.5 |
| QA4 - Two Arg. Relations | 61 | 62.9 |
| QA5 - Three Arg. Relations | 70 | 61.9 |
| QA6 - yes/No Questions | 48 | 50.7 |
| QA7 - Counting | 49 | 78.9 |
| QA8 - Lists/Sets | 45 | 77.2 |
| QA9 - Simple Negation | 64 | 64.0 |
| QA10 - Indefinite Knowledge | 44 | 47.7 |
| QA11 - Basic Coreference | 72 | 74.9 |
| QA12 - Conjunction | 74 | 76.4 |
| QA13 - Compound Coreference | 94 | 94.4 |
| QA14 - Time Reasoning | 27 | 34.8 |
| QA15 - Basic Deduction | 21 | 32.4 |
| QA16 - Basic Induction | 23 | 50.6 |
| QA17 - Positional Reasoning | 51 | 49.1 |
| QA18 - Size Reasoning | 52 | 90.8 |
| QA19 - Path Finding | 8 | 9.0 |
| QA20 - Agent's Motivations | 91 | 90.7 |

bAbI 프로젝트와 연관된 자료이다.
https://research.facebook.com/researchers/1543934539189348  

# 메모
* 기본 단어, 문장 및 쿼리 벡터 크기를 통해 GRU 모델은 다음을 달성합니다
* 20 에포크에서 QA1에 대한 52.1 % 테스트 정확도 (CPU에서 에포크 당 2 초)
* 20 에포크에서 QA2에 대한 37.0 % 테스트 정확도 (CPU에서 에포크 당 16 초) 이에 비해 Facebook 글 LSTM 기준선에 대해 50 % 및 20 %를 달성합니다
* 이 작업은 전통적으로 질문을 개별적으로 분석하지 않습니다. 이는 정확성을 향상시킬 수 있으며 두 RNN을 병합하는 좋은 예입니다.
* 단어 벡터 임베딩은 스토리와 질문 RNN간에 공유되지 않습니다.
* 1000 개가 아닌 10,000 개 훈련 샘플 (en-10k)에서 정확도가 어떻게 변경되는지 확인하십시오. 원래 논문과 비교하기 위해 1000이 사용되었습니다
* GRU, LSTM 및 JZS1-3을 사용하여 미묘하게 다른 결과를 얻을 수 있습니다.
* 길이와 노이즈 (예 : '쓸모없는'스토리 구성 요소)는 정확한 답변을 제공하는 LSTM / GRU의 기능에 영향을줍니다. 지원 사실만 고려하면 이러한 RNN은 많은 작업에서 100 % 정확도를 달성 할 수 있습니다. 주의 프로세스를 사용하는 메모리 네트워크와 신경 네트워크는이 노이즈를 효율적으로 검색하여 관련 내용을 찾아 성능을 크게 향상시킬 수 있습니다. 이는 QA1보다 훨씬 더 긴 QA2 및 QA3에서 특히 분명해집니다.

```python
from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences


def tokenize(sent):
    '''문장 부호를 포함한 문장의 토큰을 반환합니다.

    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split(r'(\W+)', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''작업 형식으로 제공되는 스토리 분석

    only_supporting이 true인경우 
    답변을 지원하는 문장만 유지됩니다.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                # 오직 연관된 substory만 선택
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # 모든 substory 제공
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''파일 이름이 주어지면 파일을 읽고 스토리를 검색 한 다음
    문장을 단일 스토리로 변환합니다.

    max_length가 제공되면, 
    max_length 토큰보다 긴 스토리는 삭제됩니다.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # index 0 은 보존되고 있다
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))

RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

try:
    path = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise

# QA1 을 1000개의 샘플로 default
# challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
with tarfile.open(path) as tar:
    train = get_stories(tar.extractfile(challenge.format('train')))
    test = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# pad_sequences를 통한 마스킹을 위해 0을 예약
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab = {}'.format(vocab))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
encoded_sentence = RNN(SENT_HIDDEN_SIZE)(encoded_sentence)

question = layers.Input(shape=(query_maxlen,), dtype='int32')
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
encoded_question = RNN(QUERY_HIDDEN_SIZE)(encoded_question)

merged = layers.concatenate([encoded_sentence, encoded_question])
preds = layers.Dense(vocab_size, activation='softmax')(merged)

model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05)

print('Evaluation')
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))
```

