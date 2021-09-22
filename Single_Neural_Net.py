import numpy as np
import matplotlib.pyplot as plt
from numpy import random as r

class uwu:
    def __init__(self):
        r.seed(1)
        self.weights = initialweight = 2*r.random((1,1)) - 1
    def tanh_deriva(self, x):
        return 1-np.tanh(x)**2
    def step(self,x):
        dot_product = np.dot(x,self.weights)
        return np.tanh(dot_product)
    def train(self, iterations, train_inputs, train_output):
        for i in range(iterations):
            output = self.step(train_inputs)
            error = train_output - output
            adjustment = np.dot(train_inputs.T,(error*self.tanh_deriva(output)))
            self.weights += adjustment

def function(x):
    return 2*x
x = [i/100 for i in range(300)]
y = [function(i/100) for i in range(300)]
data = []
for i in range(300):
    data.append(function(i/100)+r.randint(1,100)/50)
x = np.asarray([x])/100
y = np.asarray([y])/100
plt.plot(data,"b.")
plt.show()
neuron = uwu()
x = x.reshape(300,1)
y = y.T
neuron.train(10000,x,y)
constant = neuron.weights[0][0]
print(constant)
test_data = []
for i in x:
    test_data.append(i*100*constant)
plt.plot(data,"bo")
plt.plot(test_data,"r-")
plt.show()



