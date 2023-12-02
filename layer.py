import random
import numpy as np

def relu(x, deriv=False):
    return np.maximum(x, 0) if not deriv else (x >= 0).astype(int)

def sigmoid(x, deriv=False):
    return 1/(1+np.exp(-x)) if not deriv else np.exp(-x)/(1+np.exp(-x))**2

def loss(predicted, target, deriv=False):
    return np.mean((predicted - target)**2)/2 if not deriv else predicted - target

def softmax(x, deriv=False):
    x = x - np.max(x)
    softmax_values = np.exp(x)/np.sum(np.exp(x)) 
    if not deriv:
        return softmax_values
    else:
        s = softmax_values.reshape(-1, 1)
        return np.diagflat(s) - np.dot(s, s.T)

class Layer:
    def __init__(self, input_count, neuron_count, activation_fn):
        self.input_count = input_count
        self.neuron_count = neuron_count

        self.weights: np.ndarray = np.random.normal(0, 2/neuron_count, (input_count, neuron_count))
        self.biases = np.random.zeros(neuron_count)

        self.activation_fn = activation_fn

    def forward(self, inputs):
        return self.activation_fn(inputs @ self.weights + self.biases)
    
    def backward(self, inputs, output_grads, learning_rate):
        activation_derivs = self.activation_fn(inputs, deriv=True)

        output_grads *= activation_derivs

        weight_grads = np.dot(self.inputs.T, output_grads)
        bias_grads = np.sum(output_grads, axis=0)

        self.weights -= learning_rate * weight_grads
        self.biases -= learning_rate * bias_grads

        return np.dot(output_grads, self.weights.T)




class Network:
    def __init__(self, layers, loss_fn):
        self.layers = layers
        self.loss_fn = loss_fn
    
    def forward_pass(self, inputs):
        activations = [inputs]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations
    
    def compute_gradients(self, inputs, targets):
        activations = self.forward_pass(inputs)
        weight_gradients = []
        bias_gradients = []

        




    
test = Network([
    Layer(3, 4, relu),
    Layer(4, 2, relu)
], int)

print(test.forward_pass(np.array([1,2,3])))




        

    


    

