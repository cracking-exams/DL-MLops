import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=10000)

x_train = pad_sequences(x_train,maxlen=200)
x_test = pad_sequences(x_test,maxlen=200)

model = Sequential([
    layers.Embedding(input_dim=10000,output_dim=128),
    layers.GlobalAveragePooling1D(),
    
    layers.Dense(64,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(x_train,y_train,epochs=5 ,batch_size=64)
accuracy = model.evaluate(x_test,y_test)[1]

