
# from sklearn.preprocessing import MinMaxScaler
# ss = MinMaxScaler()
# X_train = ss.fit_transform(X)
# y_train = ss.fit_transform(y_true)
# y_train = y_true

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

X = np.array([7.2574, 8.3252])
X_train = X.reshape(-1, 1)
y_true = np.array([352100, 452600])
y_train = y_true.reshape(-1, 1)

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim = 1, activation = 'relu'))
model.add(Dense(1))

l = []

weight = np.array([[-21001.123, 524515.6476, -4248633.503, 11587569.35]])
bias = np.array([1.0, 1.0, 1.0, 1.0])

l.append(weight)
l.append(bias)

model.layers[0].set_weights(l)

model.compile(Adam(lr = 0.1), loss = 'mean_squared_error')
model.fit(X_train, y_train, verbose = 2, epochs = 200)

print(model.predict(X_train))
print(model.get_weights())

# print(X_train)
# print(y_train)
# temp = model.predict(X_train)
# print(temp)
# print(ss.inverse_transform(X_train))7
# print(model.get_weights())