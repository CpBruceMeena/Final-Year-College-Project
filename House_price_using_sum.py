import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()
print(df.head(10))
#df.hist(bins = 20)
#pd.plotting.scatter_matrix(df, alpha = 0.3)

df = df.loc[:10000, :]

X = df[['totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values
y_true = df[['total']].values


from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
ss = MinMaxScaler()

X= ss.fit_transform(X)
y_true = ss.fit_transform(y_true)

from keras.models import Sequential
from keras.optimizers import SGD, Adam
from keras.layers import Dense

#, activation= 'linear'
model = Sequential()
model.add(Dense(25, input_dim = 5, activation = 'sigmoid'))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true,  test_size= 0.3)

model.compile(optimizer = Adam(lr = 0.001), loss = 'mean_squared_error', metrics = ['acc'])
model.fit(X_train, y_train, verbose = 2, epochs = 10)

#X_test = X_test[:50, :]
#y_test = y_test[:50, :]


score = model.evaluate(X_test, y_test)
print(score)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

plt.plot(y_test_pred[:50, :], color = 'red', marker = 'x', label = 'y prediction')
plt.plot(y_test[:50, :], color = 'green', label = 'y original')
plt.legend(loc = 'upper left')
plt.title("Using the sum as the target variable")
plt.show()

from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))
