import tensorflow as tf
import numpy as np

predict_model = tf.keras.models.load_model('./models/composer_model')

# 문자 집어넣어서 predict 하기 -> 당연히 전처리 해줘야함

text = open('./composer/pianoabc.txt', 'r').read()

unique_text = sorted(list(set(text)))

text_to_num = {}
num_to_text = {}
for i, data in enumerate(unique_text):
    text_to_num[data] = i
    num_to_text[i] = data

number_text = []
for i in text:
    number_text.append(text_to_num[i])

arbitrary_data = number_text[222:222+25]
prepro_data = tf.one_hot(arbitrary_data, len(unique_text))
prepro_data = tf.expand_dims(prepro_data, axis=0)

# 악보 만들기 
# 1. predict()로 다음문자 예측
# 2. 예측한 다음문자 [] 저장
# 3. []에 예측값 append
# 4. 새로운 입력값[]으로 predict 하기
# 5. 반복

music = ""
for i in range(200):
    pred = predict_model.predict(prepro_data)
    old_pred = np.argmax(pred[0]) # 확률 최댓값만 뽑음
    # 변칙성, 랜덤성 주는법 -> 최댓값을 뽑는게 아니라 확률대로 뽑기
    new_pred = np.random.choice(unique_text, 1, p=pred[0])# (어디서, 몇개를, 어느확률대로 뽑을지)
    # print(new_pred)

    music += new_pred[0]
    next_data = prepro_data.numpy()[0][1:] # 맨 앞 data(one hot encoded) 짜르기
    one_hot_num = tf.one_hot(text_to_num[new_pred[0]], len(unique_text))

    # 복잡한 자료구조 다룰때 (리스트 안에 리스트 등등...) -> np.vstack() 사용시 쉽게 행렬 합치기 가능
    prepro_data = np.vstack([next_data, one_hot_num.numpy()]) 
    prepro_data = tf.expand_dims(prepro_data, axis=0)

# for i in range(200):
#     # data 전처리
#     prepro_data = tf.one_hot(arbitrary_data, len(unique_text))
#     # data shape 맞추기!
#     prepro_data = tf.expand_dims(prepro_data, axis=0)
#     # print(arbitrary_data)
#     pred = predict_model.predict(prepro_data)
#     new_pred = np.random.choice(unique_text, 1, p=pred[0])
#     print(new_pred)
#     music += new_pred[0]
#     arbitrary_data.append(text_to_num[new_pred[0]])
#     arbitrary_data[1:len(arbitrary_data)+1]
    
print(music)
