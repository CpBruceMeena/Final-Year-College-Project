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

plt.scatter(X[:, 0], y_true)
plt.show()

plt.scatter(X[:, 1], y_true)
plt.show()

plt.scatter(X[:, 2], y_true)
plt.show()

plt.scatter(X[:, 3], y_true)
plt.show()

plt.scatter(X[:, 4], y_true)
plt.show()

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
ss = MinMaxScaler()

X = ss.fit_transform(X)
y_true = ss.fit_transform(y_true)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true,  test_size= 0.3, shuffle = True)

X_test = X_test[:100, :]
y_test = y_test[:100, :]

from sklearn.svm import SVR

svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, gamma='auto')
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=3, epsilon=.1,
               coef0=1)

svr_lin.fit(X_train, y_train)
y_pred = svr_lin.predict(X_test)

y_test_pred = pd.DataFrame(y_pred, columns= ['predictions'])
y_test_pred = y_test_pred[['predictions']].values

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_pred_plot = y_test_pred[:50, :]
y_test_plot = y_test[:50, :]

plt.plot(y_test_plot, color = 'green', label = 'y_original',marker = 'o')
plt.plot(y_pred_plot, color = 'red', label = 'y_predicted', marker = 'x')
plt.title("Using SVR linear")
plt.legend(loc = 'upper left')
plt.show()

y_train_pred = svr_lin.predict(X_train)
y_train_pred = pd.DataFrame(y_train_pred, columns= ['predictions'])
y_train_pred = y_train_pred[['predictions']].values

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)
print(svr_lin.score(X_test, y_test))
print("the r2 score for the linear svr case")
from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))

y_train = ss.fit_transform(y_train)
y_test = ss.fit_transform(y_test)

svr_rbf.fit(X_train, y_train)
y_pred = svr_rbf.predict(X_test)

y_test_pred = pd.DataFrame(y_pred, columns= ['predictions'])
y_test_pred = y_test_pred[['predictions']].values

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_pred_plot = y_test_pred[:50, :]
y_test_plot = y_test[:50, :]

plt.plot(y_test_plot, color = 'green', label = 'y_original',marker = 'o')
plt.plot(y_pred_plot, color = 'red', label = 'y_predicted', marker = 'x')
plt.legend(loc = 'upper left')
plt.title("Using SVR RBF")
plt.show()

y_train_pred = svr_rbf.predict(X_train)

y_train_pred = pd.DataFrame(y_train_pred, columns= ['predictions'])
y_train_pred = y_train_pred[['predictions']].values

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)

print(svr_rbf.score(X_test, y_test))
print("the r2 score for the RBF svr case")
from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))

y_train = ss.fit_transform(y_train)
y_test = ss.fit_transform(y_test)

svr_poly.fit(X_train, y_train)
y_pred = svr_poly.predict(X_test)

y_test_pred = pd.DataFrame(y_pred, columns= ['predictions'])
y_test_pred = y_test_pred[['predictions']].values

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_pred_plot = y_test_pred[:50, :]
y_test_plot = y_test[:50, :]

plt.plot(y_test_plot, color = 'green', label = 'y_original',marker = 'o')
plt.plot(y_pred_plot, color = 'red', label = 'y_predicted', marker = 'x')
plt.title("Using SVR Polynomial")
plt.legend(loc = 'upper left')
plt.show()

y_train_pred = svr_poly.predict(X_train)

y_train_pred = pd.DataFrame(y_train_pred, columns= ['predictions'])
y_train_pred = y_train_pred[['predictions']].values

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)

print(svr_poly.score(X_test, y_test))
print("the r2 score for the polynomial svr case")
from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))