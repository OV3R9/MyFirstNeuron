import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def Sigmoid(x):
    x = np.clip(x, -709, 709)
    return (1 / (1 + np.exp(-x)))

def Sigmoid_Derivative(x):
    return (Sigmoid(x) * (1 - Sigmoid(x)))

def Initialize():
    weight = np.random.rand()
    bias = np.random.rand()
    return weight, bias

def Forward_Propagation(x, weight, bias):
    z = weight * x + bias
    probability_output = Sigmoid(z)
    return z, probability_output

def Backward_Propagation(x, y, weight, bias, z, probability_output, learing_rate):
    error = probability_output - y

    dz = error * Sigmoid_Derivative(z)

    weight -= dz * learing_rate * x
    bias -= dz * learing_rate
    return weight, bias

def Train(x_data, y_data, weight, bias, epochs=500, learing_rate=1.2):
    for _ in range(epochs):
        for x,y in zip(x_data, y_data):
            z, probability_output = Forward_Propagation(x, weight, bias)
            weight, bias = Backward_Propagation(x, y, weight, bias, z, probability_output, learing_rate)

    return weight, bias

def Predict(x_tab, weight, bias):
    answrs = []
    for x in x_tab:
        _, probability_output = Forward_Propagation(x, weight, bias)
        answrs.append([1, probability_output]) if probability_output >= 0.5 else answrs.append([0, probability_output])
    print(answrs)


x_data = [0.1, 0.3, 0.6, 1, 0.9, 10, -10] 
y_data = [0, 0, 1, 1, 1, 1, 0]

weight, bias = Initialize()
weight, bias = Train(x_data, y_data, weight, bias)

test_data = [0.2, 0.3, 0.7, 0, 1, 0.1, 100, -50] # Detecting Nums larger than 0.5

Predict(test_data, weight, bias)


