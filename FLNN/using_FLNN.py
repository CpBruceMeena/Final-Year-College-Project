import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()
print(df.head(10))

df = df.loc[:50, :]

print(df.shape)
X = df[['medianincome']].values

y_true = df[['medianHouseValue']].values

print(X)
print(y_true)

def Tx(a):
    Tx_values = [0, 0, 0]
    Tx_values[0] = 1
    Tx_values[1] = a
    Tx_values[2] = 2*((a)**2)-1
    return Tx_values


# Weights = [0.5, 0.5, 0.5]

def output(Weights, Tx_values):
    calc_output = 0
    for i in range(len(Weights)):
        calc_output += Tx_values[i]*Weights[i]
    return calc_output

def model(X_train, y_train, epochs, learning_rate):
    n = len(X_train)
    m = [0, 0, 0]
    v = [0, 0, 0]
    bias_m = 0
    bias_v = 0
    epsilon = 0.00000001
    # bias = 1.0
    beta_1 = 0.9
    beta_2 = 0.999
    Weights = [0.5, 0.5, 0.5]
    bias = 1
    for no_of_epochs in range(epochs):
        gradient = [0, 0, 0]
        bias_gradient = 0
        for i in range(len(X_train)):
            Tx_values = Tx(X_train[i])
            calc_output = output(Weights, Tx_values)+bias
            for j in range(len(gradient)):
                gradient[j] += (1/n)*(y_train[i] - calc_output)*(Tx_values[j])
                bias_gradient +=  (1/n)*(y_train[i] - calc_output)

        bias_m = beta_1 * bias_m + (1 - beta_1) * (bias_gradient)
        bias_v = beta_2 * bias_v + (1 - beta_2) * (bias_gradient ** 2)

        bias_m_cap = bias_m / (1 - beta_1 ** (no_of_epochs + 1))
        bias_v_cap = bias_v / (1 - beta_2 ** (no_of_epochs + 1))

        bias = bias - (learning_rate) * (bias_m_cap) / ((bias_v_cap) ** (0.5) + epsilon)

        for j in range(len(Weights)):
            m[j] = beta_1*m[j] + (1 - beta_1)*(gradient[j])
            v[j] = beta_2*v[j] + (1 - beta_2)*((gradient[j])**2)

            bias_m = beta_1 * bias_m + (1 - beta_1) * (bias_gradient)
            bias_v = beta_2 * bias_v + (1 - beta_2) * (bias_gradient** 2)

            m_cap = m[j]/(1 - beta_1**(no_of_epochs + 1))
            v_cap = v[j]/(1 - beta_2**(no_of_epochs + 1))


            Weights[j] = Weights[j] - (learning_rate)*(m_cap)/((v_cap)**(0.5) + epsilon)

    return Weights, bias


