import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time

(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

trainX = trainX / 255
testX = testX / 255

# input_shape (28, 28, 1)로 reshape
trainX = trainX.reshape((trainX.shape[0], trainX.shape[1], trainX.shape[2], 1))
testX = testX.reshape((testX.shape[0], testX.shape[1], testX.shape[2], 1))

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# 원래는 tf.keras.models.Sequential() API를 사용 -> 그냥 흘러가는대로 레이어 설정
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), 
])
# sigmoid: 0~1 압축 but binary(ex - 합/불) 예측문제에 사용 -> 마지막 레이어 노드 수는 1개
# softmax: 0~1 압축 but category 예측문제에 사용 -> 카테고리 별 모든확률의 합 = 1
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


# 실행시 모델을 미리보기 가능
tf.keras.utils.plot_model(model, to_file='model_Seq.png', show_shapes=True, show_layer_names=True)

# *** Funtional API로 직접 제작 -> 레이들 합치고 섞고 등등 가능... -> 효과가 더 좋으면 채택
# 레이어를 변수에 넣기
input_ = tf.keras.layers.Input(shape=(28, 28, 1))
conv2d1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_) # 새로운 괄호에 그전 레이어를 파라미터로
maxpool1 = tf.keras.layers.MaxPooling2D((2, 2))(conv2d1)
flatten1 = tf.keras.layers.Flatten()(maxpool1)
dense1 = tf.keras.layers.Dense(28*28, activation='relu')(flatten1)
reshape1 = tf.keras.layers.Reshape((28, 28, 1))(dense1)
# input_ -> conv2d1 -> maxpoll1 -> flatten1 -> dense1 -> reshpae1 레이어 순서

concat1 = tf.keras.layers.Concatenate()([input_, reshape1])
flatten2 = tf.keras.layers.Flatten()(concat1)
output = tf.keras.layers.Dense(10, activation='softmax')(flatten2)
# input_ + reshape1 -> concat1 -> flatten2 -> output 레이어 순서

model2 = tf.keras.Model(input_, output)
model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()

tf.keras.utils.plot_model(model2, to_file='model_funcAPI.png', show_shapes=True, show_layer_names=True)
