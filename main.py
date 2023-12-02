from mnist import MNIST
import numpy as np
from data_loader import load_data

if __name__=='__main__':
    train_images, train_labels, test_images, test_labels = load_data()
    print(train_images.shape)