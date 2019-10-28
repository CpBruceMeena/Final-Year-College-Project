
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("F://PycharmProjects//Zero_to_deep_learning//cal_housing.csv")
df.info()
print(df.head(10))

df = df.loc[:10000, :]

print(df.shape)
#X = df[['housingMedianAge', 'totalRooms', 'totalBedrooms', 'population', 'households', 'medianincome']].values
X = df[['medianincome']].values

y_true = df[['medianHouseValue']].values

X_temp = []
y_temp = []

for i in range(len(X)):
    X_temp.append(X[i][0])
    y_temp.append(y_true[i][0])

my_dict = {}

for i in range(len(X)):
    my_dict[X_temp[i]] = y_temp[i]

X_temp = []
y_temp = []
for i in sorted(my_dict):
    X_temp.append(i)
    y_temp.append(my_dict[i])

plt.scatter(X_temp[:100], y_temp[:100])
for i in range(100):
    print(X_temp[i], y_temp[i])

plt.show()