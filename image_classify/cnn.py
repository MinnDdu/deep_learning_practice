import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# image를 구분하는 모델을 제작해보자
# 이미지를 어떻게 뉴럴 네트워크에 넣을까? -> 뉴럴 네트워크에는 무조건 '수'만 들어갈 수 있음 (이미지 글자 파일 불가)
# 이미지는 수많은 픽셀 단위로 이루어져 있음 -> 픽셀을 숫자로 표현 (RGB -> [0~255, 0~255, 0~255], 흑백은 0~255)

# 구글이 기본적으로 호스팅 해주는 데이터셋
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
# tuple 형으로 나옴 (사실 numpy array) -> ((trainX, trainY), (testX, testY)) -> ((이미지, 정답), (테스트용X, 테스트용Y))

# 이미지 데이터를 전처리 할때 0~255가 아니라 0~1로 미리 압축해서 넣으면 더 좋은 결과를 도출 할 수도 있음
trainX = trainX / 255
testX = testX / 255

# input_shape (28, 28, 1)로 reshape
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2], 1))



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    # Convolution layer - (n개 다른 feature - 32개 새로운 이미지, (m, m)의 커널 사이즈, padding='same'사이즈 줄지 않고 같게, activation func, input_size, ...)
    # relu는 음수가 모두 0으로 됨 -> rgb 값은 모두 0~255
    # input_size 관련 흔한 에러 -> ndim 에러: Conv2D는 4차원 데이터 필요 (a, b, c, d) input_size 에는 '하나의 데이터의 shape' 필요 즉, (b, c, d)가 a개 있음
    # 원래 우리 데이터는 28x28 행렬 -> (28, 28)의 shape 이었음 -> Conv2D에서 사용하려면 하나의 데이터가 3차원이어야함 -> (28, 28, 1)로 늘려준 것!
    # color 사진이면 RGB 이므로 (28, 28, 3) 이 되어야겠지?
    # [[0,0,...0], [0,0,...0], ...] -> [[[0],[0],...[0]], [[0],[0],...[0]]...]

    # 단순 convolution의 문제점 - 다른 이미지에선 feature 들이 다른 곳에 위치해 있다면? (ex - 자동차의 바퀴 feature가 다른 사진에선 차의 위치, 차종이 다름에 따라 위치가 다를것)
    # 해결책? - Pooling layer (Down Sampling, Up Sampling ...)
    # *** Down Sampleing Layer ***
    # 이미지를 축소 - but 단순 축소가 아닌 이미지의 중요한 부분들을 모아줌 (ex - Max Pooling - 4x4 이미지를 2x2로 줄이는데 영역별 최댓값만 가져와서 만드는 것)
    tf.keras.layers.MaxPooling2D((2,2)),
 
    # conv, pooling layer -> 이미지 특징추출, 특징을 가운데로 모아줌 -> translation invariance 하다 == 이미지 위치에 따른 문제 해결

    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    # conv, pool layer 한세트로 여러번 가능

    tf.keras.layers.Flatten(),
    # Dense 레이어는 Flatten 레이어 다음에 옴
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), 
])
# sigmoid: 0~1 압축 but binary(ex - 합/불) 예측문제에 사용 -> 마지막 레이어 노드 수는 1개
# softmax: 0~1 압축 but category 예측문제에 사용 -> 카테고리 별 모든확률의 합 = 1

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 'sparse_categorical_crossentropy' -> 정답이 정수로 라벨되어있을때 사용 Ex) [0, 3, 1, 2]
# 'categorical_crossentropy' -> 정답이 "One Hot Encoding"이 되어있을때 사용 Ex) [[1,0,0,0], [0,0,0,1], [0,1,0,0], [0,1,0,0]]


# model outline 훑어보기 -> 내가 모델 어떻게 짯는지 보는 법 -> 첫번째 layer에 input_shape (데이터 하나의 shape) 명시필요
model.summary()
# Ex) layer지나고의 Output Shape (None, 28, 10) -> 10개짜리 리스트가 28개 있고 그게 None(추후에 숫자 들어갈것) 만큼 있다 (28x10 행렬 None개)
# 내가 원하는 결과는 그냥 10개짜리 리스트 ... -> 마지막 레이어 전에 Flatten(): 1d array로 압축
# Param # : layer별 학습 가능한 w, b(bias)의 개수


# 학습 후 모델 평가하기 -> evaluate(testX, testY) -> 학습할떄 사용한 데이터셋 (trainX, trainY) 사용X -> 모델이 답을 외울 수도 있음... 
# score = model.evaluate(testX, testY)
# print(score)

# 제일 중요한 요소 evaluate()의 결과 -> 마지막 epoch의 결과와 evaluate의 accuracy의 차이가? -> 'overfitting' 현상
# overfitting 현상? -> 모델이 training 데이터셋을 반복학습 (epoch 증가)에 따라 외웠기에 정확도가 높아졌던 것!
# epoch 1회 끝날때마다 평가를 해줘보면? -> model.fit(validation_data=(validX, validY)) 
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3)
# 매 epoch 마다 val_loss, val_accuracy 평가해줌
# epoch가 매우 많아질수록 overfitting 현상 나타날 것 -> val_accuracy보고 overfitting으로 인한 거품 accuracy 증가 파악 가능 
# (accuracy는 증가하지만 val_accuracy가 더이상 잘 안오를때)


# 작명 관습 -> trainX, trainY 학습할떄 데이터셋 / validX, validY validation_data용 데이터셋 / testX, testY 맨 마지막에 한번 테스트 할때 데이터셋