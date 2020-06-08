

# Train an Auxiliary Classifier Generative Adversarial Network (ACGAN) on the MNIST dataset.

자세한 정보는 https://arxiv.org/abs/1610.09585  

~ 5 에포크 이후 합리적인 이미지가 나타나고 ~ 15 에포크까지 좋은 이미지가 나타나기 
시작해야합니다. 합성곱이 연산이 많은 작업은 CPU에서 매우 느리기 때문에 GPU를 
사용해야합니다. Theano를 사용시 컴파일 시간이 차단 될 수 있으므로 반복시, 텐서플로우
 백엔드를 사용하는것을 추천합니다.

Timings:
Hardware           | Backend | Time / Epoch
------------------:|--------:|---------------:
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min

자세한 정보 및 출력 예시는 https://github.com/lukedeo/keras-acgan 를 참조하십시오.

```python
from __future__ import print_function

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

from keras.datasets import mnist
from keras import layers
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, Dropout
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2DTranspose, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils.generic_utils import Progbar
import numpy as np

np.random.seed(1337)
num_classes = 10


def build_generator(latent_size):
    # 한 쌍의 (z, L)을 매핑합니다. 여기서 z는 잠재 벡터이고, L은 
    # P_c에서 이미지 공간 (...,28,28,1) 로 그려진 레이블.
    cnn = Sequential()

    cnn.add(Dense(3 * 3 * 384, input_dim=latent_size, activation='relu'))
    cnn.add(Reshape((3, 3, 384)))

    # (7,7,...)로 업 샘플링
    cnn.add(Conv2DTranspose(192, 5, strides=1, padding='valid',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # (14, 14, ...)로 업 샘플링
    cnn.add(Conv2DTranspose(96, 5, strides=2, padding='same',
                            activation='relu',
                            kernel_initializer='glorot_normal'))
    cnn.add(BatchNormalization())

    # (28, 28, ...)로 업 샘플링
    cnn.add(Conv2DTranspose(1, 5, strides=2, padding='same',
                            activation='tanh',
                            kernel_initializer='glorot_normal'))

    # 이것은 GAN 논문에서 일반적으로 참조되는 z 공간
    latent = Input(shape=(latent_size, ))

    # 이것이 우리의 라벨이 될 것
    image_class = Input(shape=(1,), dtype='int32')

    cls = Flatten()(Embedding(num_classes, latent_size,
                              embeddings_initializer='glorot_normal')(image_class))

    # z-공간과 조건부 임베딩 클래스 간의 아다마르 곱
    h = layers.multiply([latent, cls])

    fake_image = cnn(h)

    return Model([latent, image_class], fake_image)


def build_discriminator():
    # 참고한 논문에서 제안한 LeakyReLU를 사용하여 
    # relatively standart conv net을 .
    cnn = Sequential()

    cnn.add(Conv2D(32, 3, padding='same', strides=2,
                   input_shape=(28, 28, 1)))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(64, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(128, 3, padding='same', strides=2))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Conv2D(256, 3, padding='same', strides=1))
    cnn.add(LeakyReLU(0.2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(28, 28, 1))

    features = cnn(image)

    # 첫 번째 출력(name=generation)은 판별자가 표시중인 이미지가 가짜라고 
    # 생각하는지 여부이고, 두번째 출력(name=auxiliary)는 판별자가 이미지가
    # 속해있다고 생각하는 클래스
    fake = Dense(1, activation='sigmoid', name='generation')(features)
    aux = Dense(num_classes, activation='softmax', name='auxiliary')(features)

    return Model(image, [fake, aux])

if __name__ == '__main__':

    # 논문에서의 배치 사이즈와, 잠재 크기
    epochs = 100
    batch_size = 100
    latent_size = 100

    # https://arxiv.org/abs/1511.06434 에서 제안된 Adam의 매개변수.
    adam_lr = 0.0002
    adam_beta_1 = 0.5

    # 판별자를 구축.
    print('Discriminator model:')
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    discriminator.summary()

    # 생성자를 구축.
    generator = build_generator(latent_size)

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # 가짜 이미지를 가져오기
    fake = generator([latent, image_class])

    # 결합된 모델에서만 생성자가 학습하는것을 원함
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model([latent, image_class], [fake, aux])

    print('Combined model:')
    combined.compile(
        optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=['binary_crossentropy', 'sparse_categorical_crossentropy']
    )
    combined.summary()

    # mnist 데이터를 가지고 와서, [-1,1] 범위의 (...,28,28,1) 형태로 만
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.astype(np.float32) - 127.5) / 127.5
    x_train = np.expand_dims(x_train, axis=-1)

    x_test = (x_test.astype(np.float32) - 127.5) / 127.5
    x_test = np.expand_dims(x_test, axis=-1)

    num_train, num_test = x_train.shape[0], x_test.shape[0]

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    for epoch in range(1, epochs + 1):
        print('Epoch {}/{}'.format(epoch, epochs))

        num_batches = int(x_train.shape[0] / batch_size)
        progress_bar = Progbar(target=num_batches)


        disc_sample_weight = [np.ones(2 * batch_size),
                              np.concatenate((np.ones(batch_size) * 2,
                                              np.zeros(batch_size)))]

        epoch_gen_loss = []
        epoch_disc_loss = []

        for index in range(num_batches):
            # 노이즈의 새로운 배치를 생성.
            noise = np.random.uniform(-1, 1, (batch_size, latent_size))

            # 실제 이미지의 배치를 가져옵니다.
            image_batch = x_train[index * batch_size:(index + 1) * batch_size]
            label_batch = y_train[index * batch_size:(index + 1) * batch_size]

            # p_c에서 가져온 샘플 레이블
            sampled_labels = np.random.randint(0, num_classes, batch_size)

            # 생성 된 레이블을 컨디셔너로 사용하여 가짜 이미지 배치를 생성합니다.
            # 샘플링 된 레이블을 (len(image_batch), 1)로 변경하여 길이를 1 시퀀스로 
            # 임베딩 레이어에 넣을 수 있습니다.
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            x = np.concatenate((image_batch, generated_images))

            # 한방향의 soft real/fake 레이블 사용
            # Salimans et al., 2016
            # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
            soft_zero, soft_one = 0, 0.95
            y = np.array([soft_one] * batch_size + [soft_zero] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)
            
            # 우리는 판별자가 생성 된 이미지에서 보조 분류기의 분류 정확도를 최대화하지
            # 않기를 원하므로, 판별자가 생성 된 이미지에 대한 클래스 레이블을 생성하도록 
            # 훈련시키지 않습니다(https://openreview.net/forum?id=rJXTf9Bxg 참조).
            # 보조 분류기의 샘플 가중치 합계를 유지하기 위해 실제 이미지에 2의 샘플
            # 가중치를 할당합니다.

            # 판별자가 스스로 알아낼 수 있는지 확인
            epoch_disc_loss.append(discriminator.train_on_batch(
                x, [y, aux_y], sample_weight=disc_sample_weight))

            # 새로운 노이즈를 만든다. 우리는 여기에서 2*(배치사이즈) 크기를 생성하여
            # 생성자가 판별자와 동일한 수의 이미지를 최적화시킴
            noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, num_classes, 2 * batch_size)

            # 우리의 목적은 생성자에 대해 fake, not-fake레이블에 대해 not-fake라고 하도록,
            # 제너레이터가 판별자를 속이도록 훈련하는 것
            trick = np.ones(2 * batch_size) * soft_one

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))],
                [trick, sampled_labels]))

            progress_bar.update(index + 1)

        print('Testing for epoch {}:'.format(epoch))

        # 여기에서 테스트의 손실을 계산.

        # 새로운 배치의 노이즈를 생성
        noise = np.random.uniform(-1, 1, (num_test, latent_size))

        # p_c에서 일부 레이블을 샘플링하고 그로부터 이미지를 생성.
        sampled_labels = np.random.randint(0, num_classes, num_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        x = np.concatenate((x_test, generated_images))
        y = np.array([1] * num_test + [0] * num_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # 판별자가 스스로 알아낼 수 있는지 확인.
        discriminator_test_loss = discriminator.evaluate(
            x, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # 새로운 노이즈 생성.
        noise = np.random.uniform(-1, 1, (2 * num_test, latent_size))
        sampled_labels = np.random.randint(0, num_classes, 2 * num_test)

        trick = np.ones(2 * num_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # 퍼포먼스의 에포크 레포트 생성.
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.4f} | {3:<5.4f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))

        # 에포크마다 가중치를 세이브.
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # 표시할 자릿수 생성.
        num_rows = 40
        noise = np.tile(np.random.uniform(-1, 1, (num_rows, latent_size)),
                        (num_classes, 1))

        sampled_labels = np.array([
            [i] * num_rows for i in range(num_classes)
        ]).reshape(-1, 1)

        # 표시할 배치를 가져옴
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # 클래스 라벨별로 정렬된 실제 이미지 준비
        real_labels = y_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes]
        indices = np.argsort(real_labels, axis=0)
        real_images = x_train[(epoch - 1) * num_rows * num_classes:
                              epoch * num_rows * num_classes][indices]

        # 생성된 이미지, whtie separator, 실제 이미지를 표시.
        img = np.concatenate(
            (generated_images,
             np.repeat(np.ones_like(x_train[:1]), num_rows, axis=0),
             real_images))

        # 그리드로 정렬
        img = (np.concatenate([r.reshape(-1, 28)
                               for r in np.split(img, 2 * num_classes + 1)
                               ], axis=-1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

    with open('acgan-history.pkl', 'wb') as f:
        pickle.dump({'train': train_history, 'test': test_history}, f)

```
