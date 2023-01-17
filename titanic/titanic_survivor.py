import pandas as pd
import tensorflow as tf
import numpy as np
import math

df = pd.read_csv('./titanic/train.csv')
# 승객의 정보를 입력하면 이 사람이 타이타닉 참사에서 생존할 확률을 구해보자


# print(df)
# 승객정보 열이 많음 -> 숫자는 그렇다쳐도 다른 문자등이 포함된 데이터는 전처리를 어떻게 해야할까... -> 정수로 치환? 원핫인코딩? 임베딩레이어 사용?
# 데이터의 열이 너무 많다? tensorflow가 제공해주는 feature column 사용해보자
# feature column에는 각 column을 어떻게 전처리 해줄지 넣어두는 역할 -> Ex) [원핫인코딩컬럼, 임베딩레이어쓰는컬럼, 그냥정수로컬럼, ...]


# 데이터가 891개인데 그 중 나이가 빈칸인게 177개... -> 나이 빈칸인 데이터 삭제시 데이터 양이 너무 적어짐
# 나이는 평균으로 채워볼까...?
age_average = round(df['Age'].mean())
em_mode = df['Embarked'].mode()

df['Age'].fillna(value=age_average, inplace=True)
df['Embarked'].fillna(value=str(em_mode), inplace=True)

# tensorflow가 제공해주는 라이브러리로 데이터셋 준비하기
answer = df.pop('Survived') # answer항목 열만 pop시킴
ds = tf.data.Dataset.from_tensor_slices((dict(df), answer)) # csv데이터를 집어넣을때 feature column 이용할 예정 -> 이런식으로 데이터셋 만들어야함 ({dict}, answer)

# feature column 설계할때는 우선 각 column을 먼저 분류해보는 것이 좋음
# columns -> PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Embarked
# 그냥 정수로 넣기 -> Parch, Fare, SibSp : numeric_column
# 뭉퉁그랴서 넣기 (Ex teenage, ...) -> age : bucketized_column
# 종류 몇개없어서 카테고리화 (원핫인코딩 위해) -> Pclass, Sex, Embarked : indicator_column
# 종류가 너무 많은 카테고리 (행렬로 변환) -> Ticket : embedding_column

# Name, PassengerId는 크게 중요하지 않을것 같아서 제외
feature_columns = []
# numeric_columns
feature_columns.append(tf.feature_column.numeric_column('Parch'))
feature_columns.append(tf.feature_column.numeric_column('Fare'))
feature_columns.append(tf.feature_column.numeric_column('SibSp'))

# bucketized_column
age = tf.feature_column.numeric_column('Age')
feature_columns.append(tf.feature_column.bucketized_column(age, boundaries=[10, 20, 30, 40, 50, 60, 70, 80, 90]))

# indicator_column
vocab = df['Sex'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Sex', vocab) # (column이름, 유니크한 문자 리스트)
feature_columns.append(tf.feature_column.indicator_column(cat)) # one hot encoded 됨
vocab = df['Pclass'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Pclass', vocab)
feature_columns.append(tf.feature_column.indicator_column(cat)) # one hot encoded 됨
vocab = df['Embarked'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', vocab) 
feature_columns.append(tf.feature_column.indicator_column(cat)) # one hot encoded 됨

# embedding_column
vocab = df['Ticket'].unique()
cat = tf.feature_column.categorical_column_with_vocabulary_list('Ticket', vocab) # (column이름, 유니크한 문자 리스트)
feature_columns.append(tf.feature_column.embedding_column(cat, dimension=16)) # 행렬로 바뀜

model = tf.keras.models.Sequential([
    tf.keras.layers.DenseFeatures(feature_columns),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid'), # 0~1 로 압축, loss:binary_crossentropy랑 세트
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# model.fit(ds, batch_size=64, epochs=20) 
# 에러나옴 -> Feature (key: Age) cannot have rank 0. Given: Tensor("sequential/dense_features/Cast:0", shape=(), dtype=float32)
# feature column 사용해서 DenseFeatures 레이어를 사용하면 데이터셋을 미리 batch_size로 나누어 주어야함...

ds_batch = ds.batch(64)
model.fit(ds_batch, batch_size=64, epochs=20) 

