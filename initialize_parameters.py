import numpy as np

def init_params(hiddenSize, visibleSize):
    r = np.sqrt(6) / np.sqrt(hiddenSize + visibleSize + 1)
    W1 = np.random.rand(hiddenSize, visibleSize) * 2 * r - r
    W2 = np.random.rand(visibleSize, hiddenSize) * 2 * r - r

    b1 = np.zeros((hiddenSize, 1))
    b2 = np.zeros((visibleSize, 1))

    theta = np.concatenate((W1.flatten(), W2.flatten(), b1.flatten(), b2.flatten()))
    return theta
