import tensorflow as tf
import numpy as np

# 악보를 문자나 숫자로 치환 -> abc notation
# 클래식 음악들 멜로디 정리 파일

text = open('./composer/pianoabc.txt', 'r').read()
# print(text)

# 문자를 숫자로 바꿔야함
# 1. Bag of words 만들기 -> 출현하는 모든 단어모음
# string을 넘버링 하기
unique_text = sorted(list(set(text)))

# 2. 문자 -> 숫자 변환 (숫자 -> 문자 변환도 만들어두면 유용)
# utilities 만들기 -> 변환함수, dictionary 등을 utilities 함수, dict라고 함
# sequence data를 만들때는 utilities 부터 만들어야함
text_to_num = {}
num_to_text = {}
for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

number_text = []
for i in text:
    number_text.append(text_to_num[i])
# print(number_text)

# 이걸로 (X, Y) 데이터 만들기 -> 어떻게?
# 'sequence to vector' model 생각해보자 -> Ex) 단어들 뒤에 무슨 단어 오는지 'I went to the library to ____'
# C D E F G A B __ -> 예측 -> C D E F G A B E __ -> 예측 -> C D E F G A B E G __ -> ....
# X data -> n개의 음표, Y data -> 바로 그 다음 음표
train_x = []
train_y = []

def data_mining(x, y, arr):
    for i in range(len(arr)-25):
        x.append(arr[i:i+25])
        y.append(arr[i+25])

data_mining(train_x, train_y, number_text)

# print(np.array(train_x).shape)
# print(np.array(train_y).shape)

# 데이터를 넣기 -> 1. 정수 그대로 넣기 2. one hot encoding 하기 (unique_text의 length 만큼 [0,0, ..., 1, ...,0]) 3. unique_text가 너무 길다싶으면 embedding layer
train_x = tf.one_hot(train_x, len(unique_text))
train_y = tf.one_hot(train_y, len(unique_text))

# LSTM layer사용하는 모델 만들기
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(100, input_shape=(25, 31)), # LSTM layer를 여러개 넣고싶다? -> return_sequence=True 하기
    tf.keras.layers.Dense(31, activation='softmax') # 카테고리 판단 문제
])

# compile
# data가 원핫인코딩이 되어있는 카테고리 판단 문제 -> loss function으로 categorical_crossentropy 사용해야함 -> 마지막 레이어 활성함수 softmax와 세트!
# data가 원핫인코딩이 안되어있는 카테고리 판단 문제 -> loss function으로 sparse_categorical_crossentropy 사용
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
model.fit(train_x, train_y, batch_size=2**7, epochs=45) # LSTM은 epochs 많이 필요
# 만약 google colab에서 epochs=30 이상 진행시 -> verbose=2 필수! -> 다운되는거 멈춤

model.save('./models/composer_model')