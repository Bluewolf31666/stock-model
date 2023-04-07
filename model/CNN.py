from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

def Model():
    CNN_model = Sequential()
    CNN_model.add(Conv1D(filters=128, kernel_size=3, activation='relu', input_shape=(10,4)))
    CNN_model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    CNN_model.add(MaxPooling1D(pool_size=2))
    CNN_model.add(Flatten())
    CNN_model.add(Dense(100, activation='relu'))
    CNN_model.add(Dense(1, activation='sigmoid'))
    CNN_model.compile(loss='mse', optimizer='adam')
    return CNN_model