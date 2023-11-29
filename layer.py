import random
import numpy as np

def relu(x, deriv=False):
    return np.maximum(x, 0) if not deriv else (x >= 0).astype(int)

def sigmoid(x, deriv=False):
    return 1/(1+np.exp(-x)) if not deriv else np.exp(-x)/(1+np.exp(-x))**2

def loss(predicted, target, deriv=False):
    return np.mean((predicted - target)**2)/2 if not deriv else predicted - target

def softmax(x, deriv=False):
    return np.exp(x)/np.sum(np.exp(x)) if not deriv else 

class Layer:
    def __init__(self, input_count, neuron_count, activation_fn):
        self.input_count = input_count
        self.neuron_count = neuron_count
        self.weights: np.ndarray = np.random.normal(0, 0.1, (input_count, neuron_count))
        self.biases = np.random.normal(0, 0.1, neuron_count)
        self.activation_fn = activation_fn

    def forward(self, inputs):
        return self.activation_fn(inputs @ self.weights + self.biases)

class Network:
    def __init__(self, layers, loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn
    
    def run(self, inputs, grads=False):
        activations = [inputs]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations
    
    def compute_gradients(self, inputs, targets):
        gradients = []


    
test = Network([
    Layer(3, 4, relu),
    Layer(4, 2, relu)
], int)

print(test.run(np.array([1,2,3])))




        

    


    

