import numpy as np
import math
a = 5
b = 10
w1 = 0.5
w2 = 0.5
w3 = 0.5
w4 = 0.5
w5 = 0.6
w6 = 0.6

target = 15

m = [0 for i in range(6)]
v = [0 for i in range(6)]

mhat = [0 for i in range(6)]
vhat = [0 for i in range(6)]

b1 = 0.9
b2 = 0.999

lr = 0.1
epsilon = pow(10, -8)
print(epsilon)

for i in range(4):
    p1 = max(0, w1*a + w3*b)
    p2 = max(0, w2*a + w4*b)

    print("the values of p1 and p2 is", p1, p2)

    y = max(0, p1*w5 + p2*w6)
    error = (target - y)**2

    print("the error is ", error)

    t1 = 2*(y - target)*(w5)*(a)
    t2 = 2*(y - target)*(w6)*(a)

    t3 = 2*(y - target)*(w5)*(b)
    t4 = 2*(y - target)*(w6)*(b)

    t5 = 2*(y - target)*(p1)
    t6 = 2*(y - target)*(p2)

    print("the value of the weights gradient are ", t1, t2, t3, t4, t5, t6)

    m[0] = m[0] + (1-b1)*(t1)
    m[1] = m[1] + (1-b1)*(t2)
    m[2] = m[2] + (1-b1)*(t3)
    m[3] = m[3] + (1-b1)*(t4)
    m[4] = m[4] + (1-b1)*(t5)
    m[5] = m[5] + (1-b1)*(t6)

    v[0] = v[0] + (1-b2)*((t1)**2)
    v[1] = v[1] + (1-b2)*((t2)**2)
    v[2] = v[2] + (1-b2)*((t3)**2)
    v[3] = v[3] + (1-b2)*((t4)**2)
    v[4] = v[4] + (1-b2)*((t5)**2)
    v[5] = v[5] + (1-b2)*((t6)**2)

    for j in range(6):
        print("the value of m ", m[j])
        print("the value of v ", v[j])


    mhat[0] = (m[0])/(1 - pow(b1, i+1))
    mhat[1] = (m[1])/(1 - pow(b1, i+1))
    mhat[2] = (m[2])/(1 - pow(b1, i+1))
    mhat[3] = (m[3])/(1 - pow(b1, i+1))
    mhat[4] = (m[4])/(1 - pow(b1, i+1))
    mhat[5] = (m[5])/(1 - pow(b1, i+1))

    vhat[0] = (v[0]) / (1 - pow(b2, i + 1))
    vhat[1] = (v[1]) / (1 - pow(b2, i + 1))
    vhat[2] = (v[2]) / (1 - pow(b2, i + 1))
    vhat[3] = (v[3]) / (1 - pow(b2, i + 1))
    vhat[4] = (v[4]) / (1 - pow(b2, i + 1))
    vhat[5] = (v[5]) / (1 - pow(b2, i + 1))

    for j in range(6):
        print("The value of the mhat is ", mhat[j])
        print("The value of the vhat is ", vhat[j])

    w1 = w1 - (lr*mhat[0])/(math.sqrt(vhat[0]) + epsilon)
    w2 = w2 - (lr*mhat[1])/(math.sqrt(vhat[1]) + epsilon)

    w3 = w3 - (lr*mhat[2])/(math.sqrt(vhat[2]) + epsilon)
    w4 = w4 - (lr*mhat[3])/(math.sqrt(vhat[3]) + epsilon)

    w5 = w5 - (lr*mhat[4])/(math.sqrt(vhat[4]) + epsilon)
    w6 = w6 - (lr*mhat[5])/(math.sqrt(vhat[5]) + epsilon)

    print("the value of the updated weights are ", w1, w2, w3, w4, w5, w6)

    print("the current value of the ouput is ", y)