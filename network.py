import numpy as np
from utils import softmax

class Layer:
    def __init__(self, input_count, neuron_count, activation_fn):
        self.input_count = input_count
        self.neuron_count = neuron_count

        self.weights: np.ndarray = np.random.normal(0, 2/neuron_count, (input_count, neuron_count))
        self.biases = np.zeros(neuron_count)

        self.activation_fn = activation_fn

    def forward(self, inputs):
        return self.activation_fn(inputs @ self.weights + self.biases)
    
    def backward(self, inputs, output_grads, learning_rate):
        activation_derivs = self.activation_fn(inputs, deriv=True)

        output_grads *= activation_derivs

        weight_grads = np.dot(inputs.T, output_grads)
        bias_grads = np.sum(output_grads, axis=0)

        self.weights -= learning_rate * weight_grads
        self.biases -= learning_rate * bias_grads

        return np.dot(output_grads, self.weights.T)

class Network:
    def __init__(self, layers):
        self.layers = layers
    
    def forward_pass(self, inputs):
        activations = [inputs]
        for layer in self.layers:
            activations.append(layer.forward(activations[-1]))
        return activations
    
    def backprop(self, inputs, targets, loss_fn, learning_rate):
        activations = self.forward_pass(inputs)
        loss = loss_fn(activations[-1], targets)
        output_grads = loss_fn(activations[-1], targets, deriv=True)

        for i in range(len(self.layers)-1, -1, -1):
            output_grads = self.layers[i].backward(activations[i], output_grads, learning_rate)

        return loss

    def validate(self, validation_inputs, validation_targets, loss_fn):
        activations = self.forward_pass(validation_inputs)
        validation_loss = loss_fn(activations[-1], validation_targets)
        return validation_loss

    def train(self, inputs, targets, validation_data, loss_fn, learning_rate=0.01, batch_size=8, epochs=100):
        validation_inputs, validation_targets = validation_data
        for i in range(epochs):
            train_loss = 0
            for j in range(0, len(inputs), batch_size):
                train_loss += sum(self.backprop(inputs[j:j+batch_size], targets[j:j+batch_size], loss_fn, learning_rate))
            train_loss /= len(inputs)

            validation_loss = self.validate(validation_inputs, validation_targets, loss_fn)

            print(f"Epoch {i+1}: Training Loss: {train_loss}, Validation Loss: {validation_loss}")

    def predict(self, inputs):
        activations = self.forward_pass(inputs)
        return np.argmax(activations[-1], axis=1)

    def evaluate(self, test_inputs, test_labels):
        predictions = self.predict(test_inputs)
        accuracy = np.mean(predictions == np.argmax(test_labels, axis=1))
        return accuracy

    def softmax_evaluate(self, test_inputs, test_labels):
        activations = self.forward_pass(test_inputs)
        predictions = softmax(activations[-1])
        accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
        return accuracy
