import tensorflow as tf
import matplotlib.pyplot as plt

# 딥러닝 모델을 저장하는 법 - 옷 분류 문제 모델을 저장해보자 -> model 만든 변수 이용
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # Flatten 레이어는 Dense 레이어 전에 와야함
    tf.keras.layers.Dense(64, activation='relu'), # relu : 음수는 다 0으로
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), 
])


# 1. 전체모델 저장 model.save(path)
# model.save('./models/test_model')
# models 폴더안에 test_model이라는 폴더가 모델임

# 1-2. 전체모델 불러오기 tf.keras.models.load_model(path)
# load_model = tf.keras.models.load_model('./models/test_model')
# load_model.summary()

# 2. w값들만 저장 - checkpoint 저장 - epoch 중간중간 checkpoint 저장가능!
callback_func = tf.keras.callbacks.ModelCheckpoint(
    filepath='./checkpoints/mnist', # -> epoch 끝날때마다 덮어쓰기
    # filepath='./checkpoints/mnist{epoch}', # -> epoch 마다 w값 따로 저장
    
    monitor='val_acc', mode='max', # -> val_acc가 최대가 되는 checkpoint만 저장

    save_weights_only=True,
    save_freq='epoch' # epoch 마다 저장
)

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(trainX, trainY, epochs=3, callbacks=[callback_func]) # model.fit() 안에 callbacks

# 2-2. w값만 저장해 놨으면 모델을 만들고 w값 (체크포인트 파일)을 로드하면 됨
model2 = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)), # Flatten 레이어는 Dense 레이어 전에 와야함
    tf.keras.layers.Dense(64, activation='relu'), # relu : 음수는 다 0으로
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'), 
])

model2.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# checkpoint load
model2.load_weights('./checkpoints/mnist')

model2.summary()
model2.evaluate(testX, testY)