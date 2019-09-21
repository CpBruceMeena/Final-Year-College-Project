from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import pandas as pd
import numpy as np

np.random.seed(1234)

x = [5.0, 10.0]
y = [15.0]

x = pd.DataFrame(list(zip([5], [10])), columns= ['a', 'b'])

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

model = Sequential()
model.add(Dense(2, input_dim= 2, activation = 'relu'))
model.add(Dense(1, activation= 'relu'))
l = []

weight = np.array([[0.5, 0.5], [0.5, 0.5]])
bias = np.array([0.0, 0.0])

l.append(weight)
l.append(bias)

model.layers[0].set_weights(l)

l = []
weight = np.array([[0.6], [0.6]])
bias = np.array([0.0])

l.append(weight)
l.append(bias)

model.layers[1].set_weights(l)

model.compile(Adam(lr = 0.1), loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(x, y, epochs = 3, verbose = 2)

answer = model.predict(x)
print(answer)