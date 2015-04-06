import numpy as np

def sig(x):
    return 1.0 / (1 + np.exp(-x))

sigmoid = np.vectorize(sig)

def activations(x, W1, W2, b1, b2):
    z2 = W1.dot(x) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = sigmoid(z3)      # this is the ouput of the autoencoder, h(x)

    return z2, a2, z3, a3

def sparse_autoencoder_cost(theta, visibleSize, hiddenSize, lamb, sparsity, beta, data):
    '''
    theta:          weights
    visibleSize:    number of input units
    hiddenSize:     number of hidden units
    lamb:           weight decay parameter, lambda
    sparsity:       the desired average activation for hidden units
    beta:           weight of sparsity penalty
    data:           N input vectors, each in R^n
    '''
    m = len(data)

    W1 = np.resize(theta[:hiddenSize*visibleSize], (hiddenSize, visibleSize))
    W2 = np.resize(theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize], (visibleSize, hiddenSize))
    b1 = np.mat(theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]).T
    b2 = np.mat(theta[2*hiddenSize*visibleSize+hiddenSize:]).T

    # Cost and gradient
    cost = 0;
    W1grad = np.zeros(W1.shape)
    W2grad = np.zeros(W2.shape)
    b1grad = np.zeros(b1.shape)
    b2grad = np.zeros(b2.shape)

    #print 'W1', W1.shape
    #print 'W2', W2.shape
    #print 'b1', b1.shape
    #print 'b2', b2.shape

    for x in data:
        x = np.mat(x).T
        # forward pass, for activations
        z2, a2, z3, a3 = activations(x, W1, W2, b1, b2)
        
        # cost
        cost += 0.5 / m * np.linalg.norm(a3 - x)**2

        # calculate gradient
        f3_p = np.multiply(a3, -a3 + 1) # derivative of sigmoid, f(x), is f'(x) = f(x)(1-f(x))
        d3 = np.multiply((a3 - x), f3_p)
        W2grad += d3.dot(a2.T) / m
        b2grad += d3 / m

        f2_p = np.multiply(a2, -a2 + 1)
        d2 = np.multiply(W2.T.dot(d3), f2_p)
        W1grad += d2.dot(x.T) / m
        b1grad += d2 / m

    # weight penalty
    cost += lamb / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    W1grad += lamb * W1
    W2grad += lamb * W2

    grad = np.concatenate((W1grad.flatten(), W2grad.flatten(), b1grad.flatten(), b2grad.flatten()))
    return (cost, grad)
