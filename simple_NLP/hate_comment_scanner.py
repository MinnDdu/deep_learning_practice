import tensorflow as tf
import pandas as pd
import numpy as np
import urllib.request

urllib.request.urlretrieve('https://raw.githubusercontent.com/bab2min/corpus/master/sentiment/naver_shopping.txt', './simple_NLP/shopping.txt')

# 자료료 맨위에 제목 된 행이 없으면 라벨링을 해줘야함
raw = pd.read_table('./simple_NLP/shopping.txt',names=['ratings', 'reviews'])
# 만든 raw 데이터프레임에 정답을 적는 column 넣기
# 1 = 선플 0 = 악플

raw['label'] = np.where(raw['ratings'] > 3, 1, 0)
# print(raw)

# 알다시피 딥러닝 모델에 넣으려면 수를 넣어야함
# 그런데 한글을 수로 치환하려면 고생이 필요함
# 일단 오타많음, 사람들이 띄어쓰기 잘 안 지킴, 어순 막 변형, 신조어, 자음의 반복(ㅋㅋㅋㅋ, ㄷㄷ), 등등...
# 그래서 한글은 영어와 달리 단어별로 치환이 쉽지 않음

# 데이터 전처리
# 1. 특수문자 제거해보기 -> 정규식 이용
raw['reviews'] = raw['reviews'].str.replace('[^ㄱ-ㅎㅏ-ㅣ가-힣0-9 ]', '', regex=True) # 한글및 숫자가 아닌애들 제거
# print(raw['reviews'])
# 2. 공백만 있는 행 제거
# print(raw.isnull().sum()) # 공백 (null, NaN) 값 있는지 체크 -> 없음

# 3. 중복데이터 제거하기
raw.drop_duplicates(subset=['reviews'], inplace=True)

# unique 데이터셋 만들기 (bag of words)
# 1. 직접 파이썬으로 만들기
unique_text = list(raw['reviews'])
unique_text = ''.join(unique_text) # 리스트 항목들사이에 '' 끼워서 하나의 문자열로 -> 여기선 그냥 모든 항목들 붙이기
unique_text = sorted(list(set(unique_text)))
# text_to_num = {}
# num_to_text = {}
# for i in range(len(unique_text)):
#     text_to_num[i] = unique_text[i]
#     num_to_text[unique_text[i]] = i


# 2. Tokenizing - 자연어를 정수로 바꾸기
# 딕셔너리등을 이용해 text_to_num ... 직접 만들기 가능 but keras는 Tokenizer() 라는 편리한 라이브러리 제공해줌
tokenizer = tf.keras.preprocessing.text.Tokenizer(char_level=True, oov_token='<OOV>') 
# char_level=True -> 캐릭터 단위로 정수 변환 / char_level=False -> 단어 단위로 정수 변환
# oov_token='' -> 새로운 데이터를 넣을때 저장되어있지 않은 데이터가 들어올때 어떤 스트링으로 대체할지 정함

tokenizer.fit_on_texts(raw['reviews'].tolist())
# print(tokenizer.index_word) # num_to_text 같은 친구
# print(tokenizer.num_words) # text_to_num 같은 친구

train_seq = tokenizer.texts_to_sequences(raw['reviews'].tolist())
# 이제 이 seq자료를 모델안에 넣어야함 -> 근데 문장별로 길이가 다름 -> 모델안에는 같은 길이의 문장만 넣을 수 있음...
# 모든 문장 길이를 맞추자 (padding) -> 가장 길이가 큰 문장의 길이로 통일? -> 좋은 생각 but 보통 여기서 약간 줄임
# pandas로 쉽게 파악 가능

# 길이를 계산해주는 열을 만들어보자
raw['length'] = raw['reviews'].str.len()
# print(raw.head()) # 맨위 5개 자료
# print(raw.describe()) # 데이터 요약
# print(len(raw[raw['length'] < 100]))

# train_seq 길이 조절
X = tf.keras.preprocessing.sequence.pad_sequences(train_seq, maxlen=100)
Y = np.array(raw['label'].tolist())

# 데이터 train/test/val 쪼개기
from sklearn.model_selection import train_test_split
train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=0.2, random_state=42) # random_state=n -> seed가 n 으로 셔플을 진행해줌 (42가 국룰)



# model 만들기
# one hot encoding? -> unique_text 수가 약 3000개 이상... -> 데이터가 one hot encoded 되면 글자하나하나가 [0,0,0, ... 0] -> 한 글자자료 길이가 약 3000개 이상,,,
# 따라서 one hot encoding은 비효율적이겠다~ 라는 생각의 흐름
# embedding layer 사용 -> 글자를 벡터로 바꿔줌

model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(len(tokenizer.word_index)+1, 128),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc']) # 악플인지 아닌지 판단 (binary) -> loss=binary_crossentropy
model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=10, batch_size=2**7)
model.summary()

model.save('./models/hate_cmt_scan_model')