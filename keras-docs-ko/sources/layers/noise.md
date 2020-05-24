<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L58)</span>

### GaussianDropout

```python
keras.layers.GaussianDropout(rate)
```

평균이 1인 가우시안 분포를 가지는 노이즈를 곱합니다.

규제화<sub>regularization</sub> 층이므로, 학습 과정 중에만 활성화됩니다.

__인자__

- __rate__: `float`. `Dropout`과 동일한 개념의 드롭 확률. 곱해지는 노이즈는 `sqrt(rate / (1 - rate))`의 표준편차를 갖습니다.

__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 `GaussianDropout`을
사용하려면 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정해야 합니다. 
`input_shape`는 `int`의 튜플로 배치 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일합니다.

__참조__

- [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
   http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf)
   
----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L14)</span>

### GaussianNoise

```python
keras.layers.GaussianNoise(stddev)
```

평균이 0인 가우시안 분포를 가지는 노이즈를 더합니다.

이는 무작위 데이터 증강<sub>augmentation</sub> 기법의 하나로 과적합<sub>overfitting</sub>을 완화하는데 유용합니다.
가우시안 노이즈(GS)는 실수 입력값을 변형할 때 사용됩니다.
규제화 층이므로, 학습 과정 중에만 활성화됩니다.

__인자__

- __stddev__: `float`. 노이즈 분포의 표준 편차<sub>standard deviation</sub>.


__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 `GaussianNoise`을
사용하려면 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정해야 합니다. 
`input_shape`는 `int`의 튜플로 배치 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일합니다.
    

----

<span style="float:right;">[[source]](https://github.com/keras-team/keras/blob/master/keras/layers/noise.py#L106)</span>

### AlphaDropout

```python
keras.layers.AlphaDropout(rate, noise_shape=None, seed=None)
```

입력에 알파 드롭아웃을 적용합니다.

알파 드롭아웃은 드롭아웃 이후에도 자기-정규화<sub>self-normalizing</sub> 특성이 유지
되도록 입력의 평균과 분산을 원래 값으로 유지하는 `Dropout`입니다.
알파 드롭아웃은 음수 포화<sub>saturation</sub> 값에서 무작위로 활성화 값을 지정하기 때문에,
Scaled Exponential Linear Unit(SELU)에서 학습이 잘 됩니다.



__인자__

- __rate__: `float`. `Dropout`과 동일한 개념의 드롭 확률. 곱해지는 노이즈는 `sqrt(rate / (1 - rate))`의 표준편차를 갖습니다.
    
- __noise_shape__: `int32`의 1차원 텐서. 무작위로 생성된 보관/삭제 플래그의 형태입니다.

- __seed__: `int`. 난수 생성에 사용할 시드.


__입력 형태__

임의의 형태입니다. 모델의 첫 번째 층으로 `AlphaDropout`을
사용하려면 키워드 인자 `input_shape`을 함께 사용하여 형태를 지정해야 합니다. 
`input_shape`는 `int`의 튜플로 배치 축을 포함하지 않습니다.

__출력 형태__

입력 형태와 동일합니다.

__참조__

- [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
  
