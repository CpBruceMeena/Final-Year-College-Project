import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
# df.info()
# print(df.head(10))

df = df.loc[:50, :]

X = df[['medianincome']].values

y_true = df[['medianHouseValue']].values


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_true, test_size=0.3, shuffle=True)

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
    Weights = np.random.randn(1,3)*np.sqrt(2/3)
    Weights = Weights[0]
    bias = 1
    for no_of_epochs in range(epochs):
        # print("This are the results for epoch  no.", no_of_epochs)
        gradient = [0, 0, 0]
        bias_gradient = 0
        for i in range(len(X_train)):
            Tx_values = Tx(X_train[i])
            # print("The tx values are ",  Tx_values)
            calc_output = output(Weights, Tx_values)+bias
            calc_output = max(0, calc_output)
            for j in range(len(gradient)):
                gradient[j] += (1/n)*(calc_output-y_train[i])*(Tx_values[j])
            bias_gradient += (1/n)*(calc_output-y_train[i])

        # print("The bias Gradient values are ", bias_gradient)
        # print("The Gradient values are ",  gradient)

        bias_m = beta_1 * bias_m + (1 - beta_1) * (bias_gradient)
        bias_v = beta_2 * bias_v + (1 - beta_2) * (bias_gradient ** 2)

        # print("The bias m value is ", bias_m)
        # print("The bias v value is " ,  bias_v)

        bias_m_cap = bias_m / (1 - beta_1**(no_of_epochs + 1))
        bias_v_cap = bias_v / (1 - beta_2**(no_of_epochs + 1))

        # print("The bias m_cap value is " ,  bias_m_cap)
        # print("The bias v_cap value is ",  bias_v_cap)

        bias = bias - (learning_rate) * (bias_m_cap) / ((bias_v_cap) ** (0.5) + epsilon)

        # print("The updated bias value is " ,  bias)

        for j in range(len(Weights)):

            # print("The weights of ",  j )

            m[j] = beta_1*m[j] + (1 - beta_1)*(gradient[j])
            v[j] = beta_2*v[j] + (1 - beta_2)*(gradient[j]**2)

            # print("the m value is ", m[j])
            # print("the v value is ",  v[j])

            m_cap = m[j]/(1 - beta_1**(no_of_epochs + 1))
            v_cap = v[j]/(1 - beta_2**(no_of_epochs + 1))

            # print("the m_cap value is " ,  m_cap)
            # print("the v_cap value is " ,  v_cap)

            Weights[j] = Weights[j] - (learning_rate)*(m_cap)/((v_cap)**(0.5) + epsilon)

    #         print("The updated weights are ",  Weights)
    #
    # print(Weights)
    # print(bias)
    return Weights, bias

epochs = 50000

learning_rate = 0.1

# X_train = [5.6431, 7.2574]
# y_train = [314300, 352100]

Weights, bias = model(X_train, y_train, epochs, learning_rate)

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
y_predicted = predict(X_train, Weights, bias)

plt.plot(y_train, color = 'green', label = 'y original', marker = 'x')
plt.plot(y_predicted, color = 'red', label = 'Y_prediction', marker = 'o')

plt.show()