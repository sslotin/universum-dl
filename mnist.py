from keras.datasets import mnist
from keras.utils import to_categorical

(X, y), _ = mnist.load_data()
X = X.reshape(60000, 784)
X = X.astype('float32')
X /= 255
y = to_categorical(y, 10)