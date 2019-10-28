"""
In this file we are using single feature i.e., median income to train the model
First we are using simple regression to get the three weights and then use those weights to optimize

The results we are getting are 

the r2 score on the test set is 0.485
the r2 score on the train set is 0.489


# C0 = -2.408  coefficient of X^3
# C1 = 2.69  coefficient of x^2
# C2 = 0.5025  coefficient of x
# c3 = 0.1557 constant

and the graph is saved as 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()

print(df.head(10))
df = df.loc[:10000, :]

print(df.shape)
X = df[['medianincome']].values

y_true = df[['medianHouseValue']].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
ss = MinMaxScaler()

X = ss.fit_transform(X)
y_true = ss.fit_transform(y_true)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, shuffle = True, test_size=0.3)

temp_X = np.array(X_train[:, 0])
temp_y_true = np.array(y_train[:, 0])
polynomial_3 = np.poly1d(np.polyfit(temp_X, temp_y_true, 3))

print(polynomial_3)

# C0 = -2.408  coefficient of X^3
# C1 = 2.69  coefficient of x^2
# C2 = 0.5025  coefficient of x
# c3 = 0.1557 constant


from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

model = Sequential()
model.add(Dense(4, input_dim = 1, activation = 'relu'))
model.add(Dense(1))

l = []

weight = np.array([[-2.408, 2.69, 0.5025, .1557]])
bias = np.array([1.0, 1.0, 1.0, 1.0])

l.append(weight)
l.append(bias)

model.layers[0].set_weights(l)

model.compile(Adam(lr = 0.001), loss = 'mean_squared_error')
model.fit(X_train, y_train, verbose = 2, epochs = 40)

print(model.get_weights())

temp_X_test = X_test[:50, :]
temp_y_test = y_test[:50, :]

score = model.evaluate(temp_X_test, temp_y_test, verbose = 2)
print(score)

temp_y_test_pred = model.predict(temp_X_test)

original_temp_y_test_pred = ss.inverse_transform(temp_y_test_pred)
original_temp_y_test = ss.inverse_transform(temp_y_test)

plt.plot(original_temp_y_test_pred, color = 'red', marker = 'o', label = 'y prediction')
plt.plot(original_temp_y_test, color = 'green', marker = 'x', label = 'y original')
plt.title("using neural network")
plt.legend(loc = 'upper left')
plt.show()


y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)

print(y_train)
print(y_test)

from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))