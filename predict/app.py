import tensorflow as tf
import pandas as pd 
import numpy as np

# 합격유무 / 영어성적 / 학점 / 지원 대학원 레벨 (1 > .. > 4) 데이터를 통해 합격 확률을 구해보자

data = pd.read_csv('predict/gpascore.csv')
# print(data)
# 데이터에 빵꾸난 부분등이 있을대가 대다수... -> 데이터 전처리 과정 필요!
print(data.isnull().sum()) # null 몇개인지 합산해서 나옴
data = data.dropna() # NaN(Not a Number)/빈칸있는 행 삭제시켜줌
# data = data.fillna(n) # 빈칸 n 으로 치환해줌
print(data['gpa']) # gpa 열 모두 나옴
# data['gpa'].max(), .min(), count() ...

data_y = data['admit'].values # admit 열의 각 value들을 리스트에 넣어줌
data_x = []
# pandas의 dataframe에서 사용 가능
for i, row in data.iterrows():
    # i에는 행 번호, row에는 행의 정보
    data_x.append([row['gre'], row['gpa'], row['rank']])



# DL model 만들기 -> Seuqeuntial() 쓰면 쉽게 hidden layers 자동으로 만들어 줌
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='tanh'), # layer안의 node 개수 (관습적으로 2의 제곱수로 배정) / 노드는 활성함수 필요 (activation)
    tf.keras.layers.Dense(128, activation='tanh'),
    tf.keras.layers.Dense(64, activation='tanh'),
    tf.keras.layers.Dense(1, activation='sigmoid'), # 마지막 레이어는 output layer! (확률 문제이므로 결과값 1개로 설정해야겠다!) / sigmoid는 결과 0~1사이로 압축
])

# model에 compile 해줘야 완성됨
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# optimizer 다양... adam 무난함
# loss function -> mean squared error, binary crossentropy, ... -> 0~1 확률 or 분류 문제 등에선 bce 많이 사용
# metrics 딥러닝 모델을 어떤 요소로 평가할건지 -> 보통 accuracy로...

# 학습시키기
# data 넣을때 python list는 안됨 -> numpy array 아니면 tf.tensor 자료형만 가능!
model.fit(np.array(data_x), np.array(data_y), epochs=1000) # x에는 학습데이터(예측에 필요한 인풋), y에는 실제정답, epochs는 몇 번 학습시킬지


# 예측하기 - 새로운 데이터를 같은 자료형으로 넣어주면 됨
pred = model.predict([[800, 3.8, 1], [750, 3.6, 1], [400, 2.5, 1]])
print(pred)

# 예측 결과가 좋을때 모델을 저장 하면 됨

# *** 훌륭한 데이터 전처리 과정, (하이퍼)파라미터 튜닝 과정 등으로 모델의 정확도 상승을 꾀하자 ***