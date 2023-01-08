import tensorflow as tf
import matplotlib.pyplot as plt

# image를 구분하는 모델을 제작해보자
# 이미지를 어떻게 뉴럴 네트워크에 넣을까? -> 뉴럴 네트워크에는 무조건 '수'만 들어갈 수 있음 (이미지 글자 파일 불가)
# 이미지는 수많은 픽셀 단위로 이루어져 있음 -> 픽셀을 숫자로 표현 (RGB -> [0~255, 0~255, 0~255], 흑백은 0~255)

# 구글이 기본적으로 호스팅 해주는 데이터셋
(trainX, trainY), (testX, testY) = tf.keras.datasets.fashion_mnist.load_data()
# tuple 형으로 나옴 -> ((trainX, trainY), (testX, testY)) -> ((이미지, 정답), (테스트용X, 테스트용Y))

print(trainX.shape) # -> (60000, 28, 28) -> 28x28 행렬이 60000개있는 리스트(한 행)
print(trainX[0])
# print(traninY) # 정답리스트 -> 라벨되어있음
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# image를 파이썬에서 띄워보기 -> matplotlib 라이브러리 이용
# plt.imshow(trainX[1])
# plt.gray() # 진짜 흑백으로 띄워줌
# plt.colorbar() # color를 수치화
# plt.show()


# keras 딥러닝 1.모델 만들기 2.complie 3.fit
# 확률예측문제 -> 마지막 아웃풋 레이어 노드 수는 카테고리 수에 맞추는게 좋음
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, input_shape=(28, 28), activation='relu'), # relu : 음수는 다 0으로
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10, activation='softmax'), 
])
# sigmoid: 0~1 압축 but binary(ex - 합/불) 예측문제에 사용 -> 마지막 레이어 노드 수는 1개
# softmax: 0~1 압축 but category 예측문제에 사용 -> 카테고리 별 모든확률의 합 = 1

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# 'sparse_categorical_crossentropy' -> 정답이 정수로 라벨되어있을때 사용 Ex) [0, 3, 1, 2]
# 'categorical_crossentropy' -> 정답이 "One Hot Encoding"이 되어있을때 사용 Ex) [[1,0,0,0], [0,0,0,1], [0,1,0,0], [0,1,0,0]]


# model outline 훑어보기 -> 내가 모델 어떻게 짯는지 보는 법 -> layer에 input_shape (데이터 하나의 shape) 명시필요
model.summary()
# Ex) layer지나고의 Output Shape (None, 28, 10) -> 10개짜리 리스트가 28개 있고 그게 None(추후에 숫자 들어갈것) 만큼 있다 (28x10 행렬 None개)
# 내가 원하는 결과는 그냥 10개짜리 리스트 ... -> 마지막 레이어 전에 Flatten(): 1d array로 압축
# Param # : layer별 학습 가능한 w, b(bias)의 개수

model.fit(trainX, trainY, epochs=1)
