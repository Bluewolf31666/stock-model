from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
def Model():

    LSTM_model = Sequential()
    LSTM_model.add(LSTM(64, activation=None, input_shape=(10,4), return_sequences = True))
    LSTM_model.add(LSTM(32, activation=None, return_sequences = True))
    LSTM_model.add(Flatten())
    LSTM_model.add(Dense(100, activation=None))
    LSTM_model.add(Dense(1, activation='sigmoid'))
    LSTM_model.compile(loss='mse', optimizer='adam')
    return(LSTM_model)