import tensorflow as tf

# let) h = 170 / f = 260 일때 키사이즈로 발사이즈 예측 모델 만들기 (linear regression)
# f = a * h + b (1차 함수)
h = 170
f = 260
a = tf.Variable(0.0)
b = tf.Variable(0.0)

# loss function
def loss_function():
    # return (실제값 - 예측값)^2 (=오차^2) -> tf.square()
    expectation = a * 170 + b
    return tf.square(260 - expectation)


# a, b 경사하강법으로 구하기
opt = tf.keras.optimizers.Adam(learning_rate=0.1) # 경사 하강법 도와주는 함수
# opt.minimize(loss function, var_list[업데이트할 모든 w값들 (a, b)])
opt.minimize(loss_function, var_list=[a, b]) # -> 경사 하강 1번 해줌 -> a, b 값 1번 
# 경사 하강 반복!
for i in range(300):
    opt.minimize(loss_function, var_list=[a, b])
    print(a.numpy(), b.numpy())

# 초기값 a, b == 0.0, 0.0 에서 a, b == 1.52, 1.52 정도로 업데이트 됨
print(1.52 * 170 + 1.52) # = 259.9 나옴 신발사이즈 실제값 260에 상당히 근접!

# 데이터가 각각 1개가 아닐때는 ?
# height = [170, 180, 175, 160]
# foot = [260, 270, 265, 255]
