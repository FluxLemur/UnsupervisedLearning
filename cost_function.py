import numpy as np

def sigmoid(x):
    np.exp(-x,x)
    np.add(1,x, x)
    np.reciprocal(x,x)
    return x

def calc_activations(x, W1, W2, b1, b2):
    z2 = W1.dot(x) + b1
    a2 = sigmoid(z2)
    z3 = W2.dot(a2) + b2
    a3 = sigmoid(z3)      # this is the ouput of the autoencoder, h(x)

    return z2, a2, z3, a3

def _KL(m1, m2):
    ''' Kullback-Leibler divergence between two Bernoulli random variables with means
        m1 and m2. For a constant m1 in [0,1], the function has a minimum when
        m2 = m1, and diverges when m2 approaches 0 or 1. Ideal for enforcing
        sparsity of unit activations.
    '''
    return m1 * np.log(m1/m2) + (1 - m1) * np.log((1 - m1) / (1 - m2))

KL = np.vectorize(_KL)

def _KL_p(m1, m2):
    ''' derivative of KL term used for weight gradients '''
    return - m1 / m2 + (1 - m1) / (1 - m2)

KL_p = np.vectorize(_KL_p)

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

    activations = np.zeros(b1.shape)
    Z2 = np.zeros((m,) + b1.shape)
    A2 = np.zeros(Z2.shape)
    Z3 = np.zeros((m,) + b2.shape)
    A3 = np.zeros(Z3.shape)

    for i in xrange(m):
        x = np.mat(data[i]).T

        # forward pass
        z2, a2, z3, a3 = calc_activations(x, W1, W2, b1, b2)
        Z2[i] = z2
        A2[i] = a2
        Z3[i] = z3
        A3[i] = a3

        activations += a2 / m

    # second loop is required because sparsity gradient term requires all activations
    for i in xrange(len(data)):
        x = np.mat(data[i]).T
        z2, a2, z3, a3 = Z2[i], A2[i], Z3[i], A3[i]
        cost += 0.5 / m * np.linalg.norm(a3 - x)**2

        # sparsity gradient term
        s_grad = KL_p(sparsity, activations)

        # calculate gradient
        f3_p = np.multiply(a3, -a3 + 1) # derivative of sigmoid, f(x), is f'(x) = f(x)(1-f(x))
        d3 = np.multiply((a3 - x), f3_p)
        W2grad += d3.dot(a2.T) / m
        b2grad += d3 / m

        f2_p = np.multiply(a2, -a2 + 1)
        d2 = np.multiply(W2.T.dot(d3) + beta * s_grad, f2_p)
        W1grad += d2.dot(x.T) / m
        b1grad += d2 / m

    # weight penalty
    cost += lamb / 2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    W1grad += lamb * W1
    W2grad += lamb * W2

    # sparsity
    s_term = np.sum(KL(sparsity, activations))
    cost += beta * s_term

    grad = np.concatenate((W1grad.flatten(), W2grad.flatten(), b1grad.flatten(), b2grad.flatten()))
    return (cost, grad)
