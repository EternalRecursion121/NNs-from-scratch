from mnist import MNIST
import numpy as np
from data_loader import load_data
from network import Network, Layer
from utils import relu, softmax, softmax_cross_entropy_loss, linear

if __name__=='__main__':
    train_images, train_labels, test_images, test_labels = load_data()

    # Split training data into training and validation sets
    split_idx = int(len(train_images) * 0.8)  # 80% for training, 20% for validation
    validation_images, validation_labels = train_images[split_idx:], train_labels[split_idx:]
    train_images, train_labels = train_images[:split_idx], train_labels[:split_idx]

    # Define the network architecture
    layers = [
        Layer(784, 128, relu),
        Layer(128, 64, relu),
        Layer(64, 10, linear)
    ]
    network = Network(layers)

    # Train the network with validation
    network.train(train_images, train_labels, (validation_images, validation_labels), softmax_cross_entropy_loss, learning_rate=0.01, batch_size=32, epochs=10)

    test_accuracy = network.softmax_evaluate(test_images, test_labels)
    print(f"Test Accuracy: {test_accuracy}")
