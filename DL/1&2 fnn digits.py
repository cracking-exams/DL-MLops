import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import  to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import numpy as np


(x_train,y_train),(x_test,y_test) = mnist.load_data()

x_train = x_train/255
x_test = x_test/255


x_test.shape

y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

model = Sequential([
    layers.Flatten(input_shape=(28,28)),
    layers.Dense(128,activation='relu'),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=5,batch_size=64)
accuracy = model.evaluate(x_test,y_test)[1]

plt.matshow(x_test[8])
y_pred = model.predict(x_test)
np.argmax(y_pred[8])