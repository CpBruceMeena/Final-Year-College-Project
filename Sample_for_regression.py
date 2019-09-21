import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

X_train = [i for i in range(1, 100)]
Y_train = [int(k**3) for k in range(1, 100)]

X_test = [i for i in range(10, 15)]
Y_test = [int(k**3) for k in range(10, 15)]

Train_data = []
Test_data = []

for i in range(len(X_train)):
    Train_data.append([X_train[i], Y_train[i]])

for i in range(len(X_test)):
    Test_data.append([X_test[i], Y_test[i]])

from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()

Train_data = ss.fit_transform(Train_data)
Test_data = ss.transform(Test_data)

X_train = Train_data[:, 0]
Y_train = Train_data[:, 1]

X_test = Test_data[:, 0]
Y_test = Test_data[:, 1]

model = Sequential()

model.add(Dense(units = 100, input_shape= (1,), activation = 'relu'))
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 100, activation = 'relu'))
model.add(Dense(units = 1, dtype= np.int))

model.compile(Adam(lr = 0.001), loss = 'mean_squared_error', metrics = ['accuracy'])
model.fit(X_train, Y_train, epochs = 100, batch_size = 16,verbose = 2)

Y_predict = model.predict(X_test)

'''
print(Y_predict)
print(Y_test)

plt.plot(Y_predict, color = 'red')
plt.plot(Y_test, color = 'green')
plt.show()
'''

New_predicted_data = []
for i in range(len(Y_predict)):
    New_predicted_data.append([X_test[i], Y_predict[i]])

Y_predict = ss.inverse_transform(New_predicted_data)[:, 1]
Y_test = ss.inverse_transform(Test_data)[:, 1]

for i in range(len(Y_predict)):
    Y_predict[i] = round(Y_predict[i])

print(Y_test)
print(Y_predict)

print(model.evaluate(Y_test, Y_predict))
#print(model.score(Y_test, Y_predict))

plt.plot(Y_predict, color = 'red')
plt.plot(Y_test, color = 'green')
plt.show()