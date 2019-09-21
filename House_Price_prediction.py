import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()
print(df.head(10))

df = df.loc[:10000, :]

print(df.shape)
#X = df[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values
X = df[['totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values

y_true = df[['medianHouseValue']].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
ss = MinMaxScaler()

X = ss.fit_transform(X)
y_true = ss.fit_transform(y_true)

from keras.models import Sequential
from keras.optimizers import SGD, Adam, Adagrad
from keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim = 5, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(50, activation = 'relu'))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true,  test_size= 0.3, shuffle = True)

model.compile(Adam(lr = 0.001), loss = 'mean_squared_error')
model.fit(X_train, y_train, verbose = 2, epochs = 40)

X_test = X_test[:50, :]
y_test = y_test[:50, :]

score = model.evaluate(X_test, y_test, verbose = 2)
print(score)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)

from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))

plt.plot(y_test_pred, color = 'red', marker = 'o', label = 'y prediction')
plt.plot(y_test, color = 'green', marker = 'x', label = 'y original')
plt.title("using neural network")
plt.legend(loc = 'upper left')
plt.show()
