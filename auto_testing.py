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

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
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
model.summary()


# *** TensorBoard 이용 -> 데이터 시각화 가능 ***
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format('CNN_once' + str(int(time.time()))))
# local 에서 띄우기 -> 터미널 
# tensorboard --logdir logs (logdir에 있는 logs 파일 열어주세요)
# 터미널에 주소가 뜸
model.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard], batch_size=128)


# 실험을 할때... 지금 모델에 Conv2D 레이어 하나를 더 추가한다면 어떨까? -> model 생성 & model.fit() 을 여러번 하기

model2 = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2,2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), 
])

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.summary()
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./logs/{}'.format('CNN_twice' + str(int(time.time()))))
model2.fit(trainX, trainY, validation_data=(testX, testY), epochs=3, callbacks=[tensorboard], batch_size=128)


# *** Early Stopping -> 더 이상 개선 없을경우 epoch 알아서 스탑 fit(callbacks=[]) 안에 넣어주면 됨***
es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# parameters
# -> monitor="val_accuracy" or "val_loss" -> 요소를 감시(관찰)
# -> patience=n -> epoch n번만큼 지나도 진전이 없으면 stop
# -> mode="max" or "min" -> val_accuracy일땐 max, val_loss 일땐 min
