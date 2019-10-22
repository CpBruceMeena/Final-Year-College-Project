import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()
print(df.head(10))

df = df.loc[:10000, :]
#X = df[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values
X = df[['totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values
y_true = df[['medianHouseValue']].values

from sklearn.preprocessing import StandardScaler, MinMaxScaler

#ss = StandardScaler()
ss = MinMaxScaler()

X = ss.fit_transform(X)
y_true = ss.fit_transform(y_true)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y_true,  test_size= 0.3, shuffle = True)

X_test = X_test[:100, :]
y_test = y_test[:100, :]

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(model.coef_)
print(model.intercept_)

y_test_pred = pd.DataFrame(y_pred, columns= ['predictions'])
y_test_pred = y_test_pred[['predictions']].values

y_test_pred = ss.inverse_transform(y_test_pred)
y_test = ss.inverse_transform(y_test)

y_pred_plot = y_test_pred[:50, :]
y_test_plot = y_test[:50,:]

y_train_pred = model.predict(X_train)

y_train_pred = pd.DataFrame(y_train_pred, columns= ['predictions'])
y_train_pred = y_train_pred[['predictions']].values

y_train_pred = ss.inverse_transform(y_train_pred)
y_train = ss.inverse_transform(y_train)

from sklearn.metrics import r2_score

print('the r2 score on the test set is {:.3f}'.format(r2_score(y_test, y_test_pred)))
print('the r2 score on the train set is {:.3f}'.format(r2_score(y_train, y_train_pred)))

plt.plot(y_test_plot, color = 'green', label = 'y original', marker = 'x')
plt.plot(y_pred_plot, color = 'red', label = 'Y_prediction', marker = 'o')
plt.legend(loc = 'upper left')
plt.title("Using Least Square Fitting")
plt.show()