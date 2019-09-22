import os
import tensorflow as tf
from keras.models import Sequential
import keras
from keras.layers import Dense, Dropout, LSTM, CuDNNLSTM
import numpy as np
from keras import backend as K
import time
import matplotlib.pyplot as plt
import sys
import itertools
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
from normalizer import norm
import pandas as pd
import random
from tokenizer import tokenize


def interp_fast(x, xp, fp=[0, 1], ext=False):  # extrapolates above range when ext is True
    interped = (((x - xp[0]) * (fp[1] - fp[0])) / (xp[1] - xp[0])) + fp[0]
    return interped if ext else min(max(min(fp), interped), max(fp))

os.chdir('C:\Git\ZSS')

df = pd.read_csv('ZSSdata.csv', header=None)

dataset = df.to_numpy()  # zss is broken, use angle_steers!
keys = ['angle_steers', 'shitty_angle', 'zss', 'output_steer', 'wheel_speeds.fl', 'wheel_speeds.fr', 'wheel_speeds.rl', 'wheel_speeds.rr']

data = [np.take(dataset, indices=i, axis=1) for i in range(dataset.shape[1])]
data_zip = list(zip(keys, data))
scales = dict(zip(keys, [[np.min(i[1]), np.max(i[1])] for i in data_zip]))

# normalize data
data_normalized = norm(data_zip, scales)

inputs = ['shitty_angle', 'output_steer', 'wheel_speeds.fl', 'wheel_speeds.fr', 'wheel_speeds.rl', 'wheel_speeds.rr']
output = 'angle_steers'  # this is the accurate tssp2 sensor

# sort data into above format with output at the end for tokenization
seq_len = 20
# if we want to predict the future steer angle, we offset the output list by a few samples and remove that amount from beginning of input lists
data_sorted = np.array([data_normalized[i] for i in inputs + [output]])  # formats dict of lists into list of lists sorted by inputs list above
data_sorted = np.stack(data_sorted, axis=1)  # join the array so that each item is a full sample
data_tokenized = tokenize(data_sorted, seq_length=seq_len + 1)  # add 1; we throw out last sample as we take output from there

# split into x_train and y_train
x_train = [np.take(i[:-1], indices=range(len(inputs)), axis=1) for i in data_tokenized]  # throw out last sample as it's the output of the model/zorro_sensor data
x_train = np.array([np.concatenate(seq).ravel() for seq in x_train])  # flatten array for 1d dense model
y_train = np.array([i[-1][-1] for i in data_tokenized])  # get last value of last sample for output

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1)

opt = keras.optimizers.Adam(lr=0.00055, decay=1e-6)
#opt = keras.optimizers.Adadelta()
#opt = keras.optimizers.RMSprop(0.001)
#opt = keras.optimizers.Adagrad()
#opt = 'adam'
#opt = 'rmsprop'
#opt = keras.optimizers.SGD(lr=0.01, decay=1e-8, momentum=0.9, nesterov=True)

layer_num = 5
nodes = 80

model = Sequential()
model.add(Dense(x_train.shape[1], activation="relu", input_shape=(x_train.shape[1:])))
for i in range(layer_num):
    model.add(Dense(nodes, activation="relu"))
model.add(Dense(1, activation='linear'))

tensorboard = TensorBoard(log_dir="logs\{}-layers-{}-nodes-{}-opt-adam-decay2".format(layer_num, nodes, 'relu'), histogram_freq=0, write_graph=True)
callbacks = [tensorboard]

model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mean_absolute_error'])
model.fit(x_train, y_train, shuffle=True, batch_size=64, validation_data=(x_test, y_test), callbacks=callbacks, epochs=80)

preds = []
for idx, i in enumerate(x_test):
    pred = model.predict([[i]])[0][0]
    pred = interp_fast(pred, [0, 1], scales['angle_steers'])
    ground = interp_fast(y_test[idx], [0, 1], scales['angle_steers'])
    preds.append(abs(pred - ground))
    #preds.append(abs(model.predict([[i]])[0][0] - y_test[idx]))
#print('Test accuracy: {}'.format(1 - sum(preds) / len(preds)), flush=True)
accuracy = round(interp_fast(sum(preds) / len(preds), [0, max(map(abs, scales['angle_steers']))], [1, 0]) * 100, 6)
print('Test accuracy: {}%'.format(accuracy), flush=True)

preds = []
for idx, i in enumerate(x_train):
    pred = model.predict([[i]])[0][0]
    pred = interp_fast(pred, [0, 1], scales['angle_steers'])
    ground = interp_fast(y_train[idx], [0, 1], scales['angle_steers'])
    preds.append(abs(pred - ground))
accuracy = round(interp_fast(sum(preds) / len(preds), [0, max(map(abs, scales['angle_steers']))], [1, 0]) * 100, 6)
print('Train accuracy: {}%'.format(accuracy), flush=True)

'''sample = 3335
ground = y_train[sample]
ground = interp_fast(ground, zorro_sensor_range)
pred=model.predict([[x_train[sample]]])[0][0]
pred = interp_fast(pred, zorro_sensor_range)
stock_sensor = interp_fast(x_train[sample][1], stock_sensor_range)
print(ground, pred, stock_sensor)'''

showed = 0
while showed < 10:
    sample = random.randint(0, len(x_test))
    if y_test[sample] > .6 or y_test[sample] < .4:
        showed += 1
        pred = model.predict([[x_test[sample]]])[0][0]
        pred = interp_fast(pred, [0, 1], scales['angle_steers'])
        stock_sensor_pos = inputs.index('shitty_angle') - len(inputs)
        stock_sensor = interp_fast(x_test[sample][stock_sensor_pos], [0, 1], scales['shitty_angle'])
        ground = interp_fast(y_test[sample], [0, 1], scales['angle_steers'])
        print('Stock sensor: {}\nTSSP-2 Sensor: {}\nPrediction: {}\n'.format(stock_sensor, ground, pred))


# for i in range(len(inputs)):
#     base = np.zeros(120)
#     print(model.predict([[base]]))

def time_importance():
    base = np.zeros(x_train.shape[1])
    base = model.predict([[base]])[0][0]
    preds = []
    for i in range(x_train.shape[1]):
        a = np.zeros(x_train.shape[1])
        np.put(a, i, 1)
        preds.append(abs(model.predict([[a]])[0][0]-base))
    
    plt.figure(1)
    plt.clf()
    plt.plot(range(x_train.shape[1]), preds, label='difference from baseline')
    plt.plot(range(x_train.shape[1]), [0]*x_train.shape[1], label='zero baseline')
    plt.title('Time importance (120 is latest in time)')
    plt.legend()
    plt.show()

def feature_importance():
    base = np.zeros(x_train.shape[1])
    base = model.predict([[base]])[0][0]
    preds = {}
    for idx, i in enumerate(inputs):
        a = np.zeros(len(inputs))
        np.put(a, idx, 1)
        a = np.concatenate([a] * seq_len).ravel()
        preds[i] = abs(model.predict([[a]])[0][0] - base)
    
    plt.figure(2)
    plt.clf()
    [plt.bar(idx, preds[i], label=i) for idx, i in enumerate(preds)]
    [plt.text(idx, preds[i]+.007, str(round(preds[i], 5)), ha='center') for idx, i in enumerate(preds)]
    plt.xticks(range(0,6), inputs)
    plt.title('Feature importance (difference from zero baseline)')
    plt.ylim(0, 1)
    plt.show()