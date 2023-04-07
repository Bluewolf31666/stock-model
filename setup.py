from preprocessing import pre
import numpy as np 
from keras import callbacks
import os

def data_setup(n_steps, n_seq,sequence):
    X, y = pre.split_sequences(sequence, n_steps)
    n_features = X.shape[2]
    X = X.reshape((len(X),n_steps, n_features))
    new_y = []
    for term in y:
        new_term = term[-1]
        new_y.append(new_term)
    return X, np.array(new_y), n_features

epochs = 100
verbosity = 2
dirx = 'D:\stock model'
os.chdir(dirx)
h5 = 'network.h5'


checkpoint = callbacks.ModelCheckpoint(h5,
                                       monitor='val_loss',
                                       verbose=0,
                                       save_best_only=True,
                                       save_weights_only=True,
                                       mode='auto',
                                       period=1
)