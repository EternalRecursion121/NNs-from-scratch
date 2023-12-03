import numpy as np

def relu(x, deriv=False):
    return np.maximum(x, 0) if not deriv else (x >= 0).astype(int)

def sigmoid(x, deriv=False):
    s = 1/(1+np.exp(-x))
    return s if not deriv else s*(1-s)

def mse_loss(predicted, target, deriv=False):
    return np.mean((predicted - target)**2)/2 if not deriv else predicted - target

def linear(x, deriv=False):
    return x if not deriv else np.ones_like(x)

def softmax(x):
    x = x - np.max(x)
    return np.exp(x)/np.sum(np.exp(x)) 

def softmax_cross_entropy_loss(predicted, target, deriv=False):
    softmax_values = softmax(predicted)
    if not deriv:
        return -np.sum(target * np.log(softmax_values))
    else:
        return softmax_values - target