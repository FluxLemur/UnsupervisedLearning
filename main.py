# Training and running the autoencoder
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from generate_data import gen_data, show_data, dim, show_x
from initialize_parameters import init_params
from cost_function import sparse_autoencoder_cost, calc_activations
from numerical_gradient import check_numerical_gradient, compute_numerical_gradient
from display_network import display_network

#########
# STEP 0: Relevant parameters
#########
visibleSize =   dim*dim     # number of input units
hiddenSize =    25          # number of hidden units
sparsity =      0.01        # desired average activation of hidden units
lamb =          0.0001      # weight decay
beta =          1           # weight of sparsity penalty term

# temporarily for testing backprop algorithm
#lamb = 0
#beta = 0


#########
# STEP 1: generate N data points
#########
N = 50
data = gen_data(N)

# and initialize network
theta = init_params(hiddenSize, visibleSize)


#########
# STEP 2: sparse autoencoder cost function
#########
J = lambda(x): sparse_autoencoder_cost(x, visibleSize, hiddenSize, lamb, sparsity, beta, data)


#########
# STEP 3: gradient checking
#########
#check_numerical_gradient()     # check that the numerical gradient approximation is accurate
#numgrad = compute_numerical_gradient(J, theta)
#cost, grad = J(theta)
#
#diff = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
#print 'gradient check should be close:', diff


#########
# STEP 4: train autoencoder with L-BFGS-B
#########
opt_theta, cost, info = fmin_l_bfgs_b(J, theta, maxiter=300)

#print 'minimum parameter value:', opt_theta
print 'minimum value of cost function:', cost
print 'number of iterations:', info['nit']


#########
# STEP 5: visualizing the network
#########
W1 = np.resize(opt_theta[0:hiddenSize*visibleSize], (hiddenSize, visibleSize))
W2 = np.resize(opt_theta[hiddenSize*visibleSize:2*hiddenSize*visibleSize], (visibleSize, hiddenSize))
b1 = np.mat(opt_theta[2*hiddenSize*visibleSize:2*hiddenSize*visibleSize+hiddenSize]).T
b2 = np.mat(opt_theta[2*hiddenSize*visibleSize+hiddenSize:]).T

#print 'parameter sizes', np.sum(np.square(W1)), np.sum(np.square(W2))

display_network(W1, len(W1), dim)


########
# STEP 6: training and test set visualization
########
def disp_results(data_set, show=-1):
    error = 0
    show_data(data_set[:show])
    x_hats = []
    for x in data_set[:show]:
        x = np.mat(x).T
        a2, a3 = calc_activations(x, W1, W2, b1, b2)
        x_hats.append(a3)
        error += np.linalg.norm(a3 - x)**2

    show_data(x_hats)
    print 'average error:', np.sqrt(error / len(x_hats))


disp_results(data, 20)
test_set = gen_data(20)
disp_results(test_set)
