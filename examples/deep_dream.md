# Deep Dreaming in Keras.

실행 방법:  
```python
python deep_dream.py path_to_your_base_image.jpg prefix_for_results
```

예시:  
```python
python deep_dream.py img/mypic.jpg results/dream
```

``` python
from __future__ import print_function

from keras.preprocessing.image import load_img, save_img, img_to_array
import numpy as np
import scipy
import argparse

from keras.applications import inception_v3
from keras import backend as K

parser = argparse.ArgumentParser(description='Deep Dreams with Keras.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')

args = parser.parse_args()
base_image_path = args.base_image_path
result_prefix = args.result_prefix

# 다음은 활성화를 최대화하려는
# 층의 이름과 최종 손실의
# 가중치 입니다.
# 이 설정을 조절하여 새로운 시각적인
# 효과를 얻을 수 있습니다.
settings = {
    'features': {
        'mixed2': 0.2,
        'mixed3': 0.5,
        'mixed4': 2.,
        'mixed5': 1.5,
    },
}


def preprocess_image(image_path):
    # 사진을 텐서로 받고
    # 크기 조절, 포맷의 기능을 하는 함수.
    img = load_img(image_path)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = inception_v3.preprocess_input(img)
    return img


def deprocess_image(x):
    # 텐서를 유요한 이미지로 변환하는 함수.
    if K.image_data_format() == 'channels_first':
        x = x.reshape((3, x.shape[2], x.shape[3]))
        x = x.transpose((1, 2, 0))
    else:
        x = x.reshape((x.shape[1], x.shape[2], 3))
    x /= 2.
    x += 0.5
    x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x

K.set_learning_phase(0)

# 플레이스홀더를 사용하여 InceptionV3 네트워크를 구성.
# 사전 학습된 ImageNet의 가중치를 모델로 불러옵니다.
model = inception_v3.InceptionV3(weights='imagenet',
                                 include_top=False)
dream = model.input
print('Model loaded.')

# 각 "key" 레이어의 상징적인 출력을 가져오기(우리는 고유한 이름을 지정했었음.)
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# 손실 정의
loss = K.variable(0.)
for layer_name in settings['features']:
    # 손실에 층의 feature 의 L2 노름을 추가
    if layer_name not in layer_dict:
        raise ValueError('Layer ' + layer_name + ' not found in model.')
    coeff = settings['features'][layer_name]
    x = layer_dict[layer_name].output
    # 손실에 경계가 아닌 픽셀만 포함시켜 인위 구조의 경계를 방지
    scaling = K.prod(K.cast(K.shape(x), 'float32'))
    if K.image_data_format() == 'channels_first':
        loss = loss + coeff * K.sum(K.square(x[:, :, 2: -2, 2: -2])) / scaling
    else:
        loss = loss + coeff * K.sum(K.square(x[:, 2: -2, 2: -2, :])) / scaling

# 손실에 대한 dream wrt의 그래디언트를 계산
grads = K.gradients(loss, dream)[0]
# 그래디언트를 정규화.
grads /= K.maximum(K.mean(K.abs(grads)), K.epsilon())

# 입력한 이미지에서 손실 및
# 그래디언트 값을 검색하는 함수를 설정.
outputs = [loss, grads]
fetch_loss_and_grads = K.function([dream], outputs)


def eval_loss_and_grads(x):
    outs = fetch_loss_and_grads([x])
    loss_value = outs[0]
    grad_values = outs[1]
    return loss_value, grad_values


def resize_img(img, size):
    img = np.copy(img)
    if K.image_data_format() == 'channels_first':
        factors = (1, 1,
                   float(size[0]) / img.shape[2],
                   float(size[1]) / img.shape[3])
    else:
        factors = (1,
                   float(size[0]) / img.shape[1],
                   float(size[1]) / img.shape[2],
                   1)
    return scipy.ndimage.zoom(img, factors, order=1)


def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = eval_loss_and_grads(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x


"""Process:
- 원본 이미지를 불러온다.
- 크기을 프로세싱 하는 수를 정의 (즉, 이미지의 형태),
    가장 작은것에서 큰 것 까지.
- 원본 이미지의 크기를 가장 작은 크기로 조정.
- 모든 크기에 대해, 가장 작은 것으로 시작 (즉, 지금의 것):
    - 경사상습법 실행
    - 다음 크기로 이미지를 업 스케일.
    - 업 스케일시 손상된 이미지를 복구.
- 원래 크기로 돌아오면 종료.
업 스케일링 중 손실 된 디테일을 얻으려면,
원본 이미지를 가져와 축소하고, 
업 스케일 한 결과를 원본 이미지와 비교하십시오.
"""

# 이 하이퍼 매개변수를 사용하면 새로운 효과를 얻을 수 있습니다.
step = 0.01  # 경사 상승법 단계 사이즈
num_octave = 3  # 그라디언트 상승을 실행할 스케일 수
octave_scale = 1.4  # 스케일 사이의 크기 비율
iterations = 20  # 스케일 당 상승 단계 수
max_loss = 10.

img = preprocess_image(base_image_path)
if K.image_data_format() == 'channels_first':
    original_shape = img.shape[2:]
else:
    original_shape = img.shape[1:3]
successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

save_img(result_prefix + '.png', deprocess_image(np.copy(img)))
```
