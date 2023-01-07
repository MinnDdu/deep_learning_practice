import tensorflow as tf

# tensor는 행렬이랑 비슷한 자료형
# tf.constant()

tensor = tf.constant([3.0, 4, 5])
tensor2 = tf.constant([4.2, 5, 5])
tensor3 = tf.constant([[1, 2], [3, 4]])
tensor4 = tf.constant([[[1, 2], [2, 3], [3, 4]], [[1, 2], [2, 3], [3, 4]]])
print(tensor + tensor2)

# tensor 자료형 공식들
# tf.add() / tf.subtract() / tf.divide() / tf.divide() / tf.mutiply / tf.matmul() - 행렬의 곱 (dot product)
print(tf.add(tensor, tensor2))
# tf.zeros(n) -> 0 이 n 개 있는 tensor 만들어줌 / tf.zeros([n, m]) -> n 행 m 열의 0이 있는 tensor 생성 / tf.zeros([p, n, m]) -> n 행 m 열을 p 개 생성
# * tensor의 shape * : 몇차원의 행렬이구나 등등 자료형 파악에 매우 중요
print(tensor.shape) # -> (3)
print(tensor3.shape) # -> (2, 2)
print(tensor4.shape) # -> (2, 3, 2)

print(tf.zeros(10))

# tf의 dtype (data type: int, float) -> 보통 float으로 많이 사용
# tf.cast() -> casting method

# tf.Variable() -> Weight 값 만들때 사용!
w = tf.Variable(1.0)
print(w)
print(w.numpy()) # -> 진짜 값만 출력
w.assign(2.0) # variable 값 수정
print(w.numpy())