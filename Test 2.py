import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train = [i for i in range(1, 3)]
Y_train = [k for k in range(1, 3)]

X_test = [i for i in range(1, 3)]
Y_test = [k for k in range(1, 3)]

maximum_value = 0
for i in Y_train:
    if i > maximum_value:
        maximum_value = i

for i in range(len(Y_train)):
    Y_train[i] = Y_train[i]

model = Sequential()

model.add(Dense(units = 2, input_shape= (1,), activation = 'linear'))
model.add(Dense(units = 1))

print(model.get_weights())

model.compile(Adam(lr = 0.05), loss = 'mean_squared_error')
model.fit(X_train, Y_train, batch_size = 32, epochs = 100, verbose = 2)

Y_predict = model.predict(X_test)

for i in range(len(Y_predict)):
    Y_predict[i] = Y_predict[i]

print(Y_test)
print(Y_predict)

plt.plot(Y_predict, color = 'red')
plt.plot(Y_test, color = 'green')
plt.show()