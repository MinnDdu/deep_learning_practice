import tensorflow as tf
import random

train_x = [1, 2, 3, 4, 5, 6, 7]
train_y = [3, 5 ,7 ,9, 11, 13, 15]

# sequence of DL
# 1. 모델 만들기 (수식)
# 보통 변수들의 초기값은 randomize함
a = tf.Variable(random.uniform(0.01, 0.5))
b = tf.Variable(random.uniform(0.01, 0.75))

# 2. 경사하강법을 도울 optimizer 고르기 -> 경사하강 학습하기 (손실함수 필요)!
def loss_function(a, b):
    # 대충 a, b이런 수식이지 않을까...? (a, b 가 tf.Variable 이라 이런 수식 가능 행렬에 상수 곱하듯이..)
    # * 예측값을 만들때 Neural Network 를 이용해서 예측값 만들기 -> Deep Learning *
    expect_y = a * train_x + b 
    # mean squared error = ((예측1 - 실제1)^2 + (예측2 - 실제2)^2 ... ) / n
    return tf.keras.losses.mse(train_y, expect_y)

# learning rate (alpha) 같은 hyperparameter(사용자가 직접세팅)는 좋은 값 찾기 위해 trial and error 필요...
opt = tf.keras.optimizers.Adam(learning_rate=0.1) 
for i in range(500):
    opt.minimize(loss_function, var_list=[a, b])
    # opt.minimize(lambda:loss_function(a, b), var_list=[a, b]) -> 람다 함수 이용도 가능
    print(a.numpy(), b.numpy())
