
"""*
# the r2 score on the test set is 0.653
# the r2 score on the train set is 0.487
# The graph has been saved by the name of using_single_feature_auto_medianIncome
"""


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()

df = df.loc[:800, :]

print(df.shape)
#X = df[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values

X = df[['medianincome']].values
y_true = df[['medianHouseValue']].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
xs = MinMaxScaler()
ys = MinMaxScaler()

# X = xs.fit_transform(X)
# y_true = ys.fit_transform(y_true)

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense

model = Sequential()
model.add(Dense(30, input_dim = 1, activation = 'relu'))
model.add(Dense(1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true, shuffle = True, test_size=0.3)

xs.fit(X_train)
X_train = xs.transform(X_train)
X_test = xs.transform(X_test)

ys.fit(y_train)
y_train = ys.transform(y_train)
y_test = ys.transform(y_test)

model.compile(Adam(lr = 0.001), loss = 'mean_squared_error')
model.fit(X_train, y_train, verbose = 2, epochs = 50)

# print(model.get_weights())

temp_X_test = xs.transform(X[:50, :])
temp_y_test = y_true[:50, :]

temp_y_test_pred = model.predict(temp_X_test)

original_temp_y_test_pred = ys.inverse_transform(temp_y_test_pred)

plt.plot(original_temp_y_test_pred, color = 'red', marker = 'o', label = 'y prediction')
plt.plot(temp_y_test, color = 'green', marker = 'x', label = 'y original')
plt.title("using neural network")
plt.legend(loc = 'upper right')
plt.show()


y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

y_test_pred = ys.inverse_transform(y_test_pred)
y_test = ys.inverse_transform(y_test)

y_train_pred = ys.inverse_transform(y_train_pred)
y_train = ys.inverse_transform(y_train)

from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))

# predicting for the whole dataset that we are choosing

X_train = X[:500, :]
X_train = xs.transform(X_train)

y_train = y_true[:500, :]

y_pred = model.predict(X_train)

y_pred = ys.inverse_transform(y_pred)

y_img_pred = y_pred[:50, :]
y_img_original = y_train[:50, :]

plt.plot(y_img_pred, color = 'red', marker = 'o', label = 'y prediction')
plt.plot(y_img_original, color = 'green', marker = 'x', label = 'y original')
plt.title("using neural network")
plt.legend(loc = 'upper left')
plt.show()

print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_pred)))