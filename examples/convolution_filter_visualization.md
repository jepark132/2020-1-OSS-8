# Visualization of the filters of VGG16, via gradient ascent in input space.


이 스크립트는 몇 분 안에 CPU에서 실행될 수 있습니다.

결과 예시:
![example](./images/ml.jpg)


``` python
from __future__ import print_function

import time
import numpy as np
from PIL import Image as pil_image
from keras.preprocessing.image import save_img
from keras import layers
from keras.applications import vgg16
from keras import backend as K


def normalize(x):
    """텐서를 정규화하는 기능의 함수.

    # Arguments
        x: 입력 텐서

    # Returns
        The 정규화된 입력 텐서.
    """
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())


def deprocess_image(x):
    """실수 배열을 유효한 uint8로 변환하는 기능의 함수.

    # Arguments
        x: 생성된 이미지를 나타내는 넘파이 배열.

    # Returns
        imshow 등에서 사용할 처리된 넘파이 배열
    """
    # 텐서 정규화 : 0의 중심, 표준이 0.25인지 확인
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.25

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # RGB 배열로 변환
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def process_image(x, former):
    """ uint8의 이미지를 다시 실수 배열로 변환하는 기능의 함수.
        'deprocess_img' 의 역

    # Arguments
        x: imshow 등에서 사용할 처리된 넘파이 배열
        former: 이전 넘파이 배열.
                이전의 평균과 분산을 결정해야 합니다.

    # Returns
        생성된 이미지를 나타내도록 처리된 넘파이 배열.
    """
    if K.image_data_format() == 'channels_first':
        x = x.transpose((2, 0, 1))
    return (x / 255 - 0.5) * 4 * former.std() + former.mean()


def visualize_layer(model,
                    layer_name,
                    step=1.,
                    epochs=15,
                    upscaling_steps=9,
                    upscaling_factor=1.2,
                    output_dim=(412, 412),
                    filter_range=(0, None)):
    """특정 모델에서 하나의 합성곱층에서 가장 관련성이 높은 필터를 시각화합니다.

    # Arguments
        model: layer_name을 포함하는 모델.
        layer_name: 시각화할 층의 이름.
                    모델의 일부여야 합니다.
        step: 경사 상습법을위한 단계 크기.
        epochs: 경사 상승법의 반복 횟수.
        upscaling_steps: 업샘플링의 횟수.
                         여기서 시작의 이미지는 (80, 80) 입니다.
                         컨펌내용 확인용 이 내용이 그대로 있으면 컨펌을 확인 하지 않은것.
        upscaling_factor: output_dim 으로 이미지를 천천히
                          업그래이드 할 요소.
        output_dim: [img_width, img_height] 출력 이미지 차원.
        filter_range: [lower, upper]의 튜플
                      계산할 필터의 수를 결정.
                      만약 두번째 값이 "None" 일 경우,
                      마지막 필터는 상한으로 간주됩니다.
    """

    def _generate_filter_image(input_img,
                               layer_output,
                               filter_index):
        """하나의 특정 필터에 대한 이미지를 생성합니다.

        # Arguments
            input_img: 입력 이미지 텐서
            layer_output: 출력 이미지 텐서
            filter_index: 처리 할 필터 수
                          유효하다고 가정.

        #Returns
            이미지를 생성 할 수 없거나,
            이미지(array) 자체와 마지막 손실의 튜플이 없으면 None입니다.
        """
        s_time = time.time()

        # 고려 된 레이어의 n 번째 필터의 활성화를 
        # 최대화하는 손실 함수를 구축합니다.
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        # 이 손실로 입력 이미지 기울기를 계산
        grads = K.gradients(loss, input_img)[0]

        # 정규화 트릭 : 그라디언트를 정규화다
        grads = normalize(grads)

        # 이 함수는 입력 그림이 주어지면 손실과 그라데이션을 반환합니다
        iterate = K.function([input_img], [loss, grads])

        # 임의의 노이즈로 회색 이미지에서 시작합니다
        intermediate_dim = tuple(
            int(x / (upscaling_factor ** upscaling_steps)) for x in output_dim)
        if K.image_data_format() == 'channels_first':
            input_img_data = np.random.random(
                (1, 3, intermediate_dim[0], intermediate_dim[1]))
        else:
            input_img_data = np.random.random(
                (1, intermediate_dim[0], intermediate_dim[1], 3))
        input_img_data = (input_img_data - 0.5) * 20 + 128

        # 원래 크기로 천천히 업 스케일하면 412d 이미지를
        # 직접 계산할 때 빈번하게 발생하는 시각적인 구조를
        # 억제할 수 있습니다. 각각의 다음 차원에 대해 더 나은
        # 시작점에서 작동하므로 좁은 극소점을 피합니다.
        for up in reversed(range(upscaling_steps)):
            # 경사상승법을 실행 e.g. 20 steps
            for _ in range(epochs):
                loss_value, grads_value = iterate([input_img_data])
                input_img_data += grads_value * step

                # 일부 필터는 0에 고정되어 건너 뛸 수 있습니다
                if loss_value <= K.epsilon():
                    return None

            # 업 스케일 된 차원 계산
            intermediate_dim = tuple(
                int(x / (upscaling_factor ** up)) for x in output_dim)
            # Upscale
            img = deprocess_image(input_img_data[0])
            img = np.array(pil_image.fromarray(img).resize(intermediate_dim,
                                                           pil_image.BICUBIC))
            input_img_data = np.expand_dims(
                process_image(img, input_img_data[0]), 0)

        # 결과 입력 이미지를 디코딩
        img = deprocess_image(input_img_data[0])
        e_time = time.time()
        print('Costs of filter {:3}: {:5.0f} ( {:4.2f}s )'.format(filter_index,
                                                                  loss_value,
                                                                  e_time - s_time))
        return img, loss_value

    def _draw_filters(filters, n=None):
        """ nxn 격자에 가장 좋은 필터를 그립니다.

        # Arguments
            filters: 처리 된 각 필터에 대해 생성 된 이미지 및 해당 손실의 목록입니다.
            n: 격자의 차원
               none 일 경우, 가능한 가장 큰 윈도우가 사용됩니다.
        """
        if n is None:
            n = int(np.floor(np.sqrt(len(filters))))

        # 손실이 가장 높은 필터는 더 잘 보인다고 가정합니다.
        # 우리는 최상위 n * n 필터 만 유지합니다.
        filters.sort(key=lambda x: x[1], reverse=True)
        filters = filters[:n * n]

        # 충분한 공간을 가진 검은 그림을 만들다
        # e.g.크기가 412 x 412 인 8 x 8 필터 사이에 5px 여백
        MARGIN = 5
        width = n * output_dim[0] + (n - 1) * MARGIN
        height = n * output_dim[1] + (n - 1) * MARGIN
        stitched_filters = np.zeros((width, height, 3), dtype='uint8')

        # 저장된 필터로 사진을 채우십시오
        for i in range(n):
            for j in range(n):
                img, _ = filters[i * n + j]
                width_margin = (output_dim[0] + MARGIN) * i
                height_margin = (output_dim[1] + MARGIN) * j
                stitched_filters[
                    width_margin: width_margin + output_dim[0],
                    height_margin: height_margin + output_dim[1], :] = img

        # 결과를 디스크에 저장
        save_img('vgg_{0:}_{1:}x{1:}.png'.format(layer_name, n), stitched_filters)

    # 입력 이미지의 자리 표시자
    assert len(model.inputs) == 1
    input_img = model.inputs[0]

    # 각 "키"레이어의 기호 출력을 얻습니다 (고유한 이름 지정).
    layer_dict = dict([(layer.name, layer) for layer in model.layers[1:]])

    output_layer = layer_dict[layer_name]
    assert isinstance(output_layer, layers.Conv2D)

    # 처리 할 필터 범위 계산
    filter_lower = filter_range[0]
    filter_upper = (filter_range[1]
                    if filter_range[1] is not None
                    else len(output_layer.get_weights()[1]))
    assert(filter_lower >= 0
           and filter_upper <= len(output_layer.get_weights()[1])
           and filter_upper > filter_lower)
    print('Compute filters {:} to {:}'.format(filter_lower, filter_upper))

    # 각 필터를 반복하고 해당 이미지를 생성
    processed_filters = []
    for f in range(filter_lower, filter_upper):
        img_loss = _generate_filter_image(input_img, output_layer.output, f)

        if img_loss is not None:
            processed_filters.append(img_loss)

    print('{} filter processed.'.format(len(processed_filters)))
    # 마지막으로 디스크에 최상의 필터를 그려 저장
    _draw_filters(processed_filters)


if __name__ == '__main__':
    # 시각화하려는 레이어의 이름
    # (keras / applications / vgg16.py의 모델 정의 참조)
    LAYER_NAME = 'block5_conv1'

    # ImageNet 가중치로 VGG16 네트워크 구축
    vgg = vgg16.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')
    vgg.summary()

    # 함수 호출 예
    visualize_layer(vgg, LAYER_NAME)
```
