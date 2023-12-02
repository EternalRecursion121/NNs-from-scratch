from mnist import MNIST
import numpy as np

def one_hot_encode(lables):
    return np.eye(10)[lables]

def load_data():
    mndata = MNIST('data')
    train_images, train_labels = mndata.load_training()
    test_images, test_labels = mndata.load_testing()

    train_images = np.array(train_images) / 255
    test_images = np.array(test_images) / 255

    train_labels = one_hot_encode(np.array(train_labels))
    test_labels = one_hot_encode(np.array(test_labels))

    train_order = np.random.permutation(len(train_images))
    test_order = np.random.permutation(len(test_images))

    train_images = train_images[train_order]
    train_labels = train_labels[train_order]

    test_images = test_images[test_order]
    test_labels = test_labels[test_order]

    return train_images, train_labels, test_images, test_labels

load_data()