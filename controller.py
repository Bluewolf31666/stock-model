import setup
import preprocessing
import yfinance
import os
from model import LSTM,CNN
from keras import callbacks


df = yfinance.download('AAPL','2012-1-1','2023-1-1')
df = df.drop(['Volume'],1).drop(['Adj Close'],1)

dataset,minmax = preprocessing.pre.normalize_data(df)
values = dataset.values

n_steps = 10
n_seq = 10000
rel_test_len = 0.1
X,y,n_features = setup.data_setup(n_steps,n_seq,values)
X = X[:-1]
y = y[1:]
X_test,y_test = X[:int(len(X)*rel_test_len)],y[:int(len(X)*rel_test_len)]
X_train,y_train = X[int(len(X)*rel_test_len):],y[int(len(X)*rel_test_len):]


LSTM_model=LSTM.Model()

checkpoint=setup.checkpoint
epochs=setup.epochs
verbosity=setup.verbosity

callback = [checkpoint]
json = 'network.json'
model_json = LSTM_model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
history = LSTM_model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=len(X_train) // 4,
                    validation_data = (X_test,y_test),
                    verbose=verbosity,
                    callbacks=callback)

CNN_model=CNN.Model()
callback = [checkpoint]
json = 'network.json'
model_json = CNN_model.to_json()
with open(json, "w") as json_file:
    json_file.write(model_json)
history2 = CNN_model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    batch_size=len(X_train) // 4,
                    validation_data = (X_test,y_test),
                    verbose=verbosity,
                    callbacks=callback)

LSTM_model.evaluate(X_test,y_test)
CNN_model.evaluate(X_test,y_test)