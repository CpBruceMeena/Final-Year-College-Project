import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

# The result for this case using 50 datapoints is
#     [-0.08229602  0.52726302 - 0.09916695 - 0.16669065][0.63815072]
# The    r2    score    on    the    training    data is 0.7707040090291927

# The result for this case using 500 datapoints is
# [ 0.25132578  0.13831302 -0.40664837 -0.37903709] [0.19723916]
# The r2 score on the training data is  0.5213129429474388

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
# df.info()
# print(df.head(10))

df = df.loc[:500, :]

# X = df[['medianincome']].values
X_train = df[['medianincome']].values

# y_true = df[['medianHouseValue']].values
y_train = df[['medianHouseValue']].values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, shuffle=True)

xs = MinMaxScaler(feature_range= (-1, 1))
xs.fit(X_train)
X_train_transformed = xs.transform(X_train)

ys = MinMaxScaler()
ys.fit(y_train)
y_train_transformed = ys.transform(y_train)

def Tx(a):
    Tx_values = [0, 0, 0, 0, 0]
    Tx_values[0] = 1
    Tx_values[1] = a+1
    Tx_values[2] = 3*(a**2) + 3*a + 1
    Tx_values[3] = 15*(a**3)+15*a**2 + 6*a + 1
    Tx_values[4] = 105*(a**4) +105*a**3 + 45*a**2 + 10*a + 1
    # Tx_values[5] = 945*a**5 + 945*a**4 + 420*a**3 + 105*a**2 + 15*a + 1
    return Tx_values

# Weights = [0.5, 0.5, 0.5]

def output(Weights, Tx_values):
    calc_output = 0
    for i in range(len(Weights)):
        calc_output += Tx_values[i]*Weights[i]
    return calc_output

def model(X_train, y_train, epochs, learning_rate):
    n = len(X_train)
    m = [0, 0, 0, 0, 0]
    v = [0, 0, 0, 0, 0]
    bias_m = 0
    bias_v = 00.
    epsilon = 0.00000001
    # bias = 1.0
    beta_1 = 0.9
    beta_2 = 0.999
    Weights = (np.random.randn(1,5))*np.sqrt(2/3)
    print(Weights)
    Weights = Weights[0]
    # Weights = [-0.02345428, 0.44043124, -0.01990337]
    bias = 1
    for no_of_epochs in range(epochs):
        gradient = [0, 0, 0, 0, 0]
        bias_gradient = 0
        for i in range(len(X_train)):
            Tx_values = Tx(X_train[i])

            calc_output = output(Weights, Tx_values)+bias
            calc_output = max(0, calc_output)
            loss = 0
            for j in range(len(gradient)):
                loss += (1/n)*((calc_output-y_train[i])**2)
                gradient[j] += (2/n)*(calc_output-y_train[i])*(Tx_values[j])
            bias_gradient += (2/n)*(calc_output-y_train[i])

        print("Epoch:" + str(no_of_epochs) + "      Loss :" + str(loss))

        bias_m = beta_1 * bias_m + (1 - beta_1) * (bias_gradient)
        bias_v = beta_2 * bias_v + (1 - beta_2) * (bias_gradient ** 2)

        bias_m_cap = bias_m / (1 - beta_1**(no_of_epochs + 1))
        bias_v_cap = bias_v / (1 - beta_2**(no_of_epochs + 1))

        bias = bias - (learning_rate) * (bias_m_cap) / ((bias_v_cap) ** (0.5) + epsilon)

        for j in range(len(Weights)):

            m[j] = beta_1*m[j] + (1 - beta_1)*(gradient[j])
            v[j] = beta_2*v[j] + (1 - beta_2)*(gradient[j]**2)

            m_cap = m[j]/(1 - beta_1**(no_of_epochs + 1))
            v_cap = v[j]/(1 - beta_2**(no_of_epochs + 1))

            Weights[j] = Weights[j] - (learning_rate)*(m_cap)/((v_cap)**(0.5) + epsilon)

    return Weights, bias

epochs = 2000

learning_rate = 0.001

# X_train = [5.6431, 7.2574]
# y_train = [314300, 352100]

start_time = time.time()

Weights, bias = model(X_train_transformed, y_train_transformed, epochs, learning_rate)

print("The time required for training is ---" + str((time.time() - start_time)) + ' seconds')

def predict(X_test, Weights, bias):
    y_predicted = []
    for i in range(len(X_test)):
        Tx_values = Tx(X_test[i])
        output = 0
        for j in range(len(Weights)):
            output += Weights[j]*Tx_values[j]
        output += bias
        output = max(0, output)
        y_predicted.append(output)
    return y_predicted

print(Weights, bias)

y_predicted = predict(X_train_transformed, Weights, bias)
y_predicted = np.asarray(y_predicted)
y_predicted = y_predicted.reshape(-1, 1)
y_predicted = ys.inverse_transform(y_predicted)
y_train = ys.inverse_transform(y_train_transformed)

#
# for i in range(len(y_train)):
#     print(y_predicted[i], y_train[i])

plt.plot(y_train[:50, 0], color = 'green', label = 'y original', marker = 'x')
plt.plot(y_predicted[:50, 0], color = 'red', label = 'Y_prediction', marker = 'o')
plt.title("using 4th degree Bessel polynomial")
plt.legend(loc = 'upper right')
plt.show()

plt.show()

from sklearn.metrics import r2_score
print("The r2 score on the training data is ", r2_score(y_train, y_predicted))