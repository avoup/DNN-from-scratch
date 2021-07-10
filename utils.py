import numpy as np
from mnist import MNIST

def load_mnist():    
    mnist = MNIST('./dataset/MNIST')
    x_train, y_train = mnist.load_training() #60000 samples
    x_test, y_test = mnist.load_testing()    #10000 samples

    x_train = np.asarray(x_train).astype(np.float32).T
    y_train = np.asarray(y_train).astype(np.int32).reshape(-1, 1).T
    x_test = np.asarray(x_test).astype(np.float32).T
    y_test = np.asarray(y_test).astype(np.int32).reshape(-1, 1).T
    return x_train, y_train, x_test, y_test
