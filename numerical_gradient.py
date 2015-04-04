import numpy as np

def check_numerical_gradient():
    ''' Checks that the numerical gradient computes properly by comparing its output
        against an analytical solution.
    '''
    def simple_quadratic_f(x):
        value = x[0]**2 + 3*x[0]*x[1]
        grad = np.zeros((2,))
        grad[0] = 2*x[0] + 3*x[1]
        grad[1] = 3*x[0]

        return value, grad

    x = np.array([4.0,10.0])
    value, grad = simple_quadratic_f(x)
    numgrad = compute_numerical_gradient(simple_quadratic_f, x)
    print 'numerical gradient:', numgrad
    print 'expected :        ', grad
    norm  = np.linalg.norm(numgrad-grad) / np.linalg.norm(numgrad+grad)
    if norm > 1e-9:
        print 'norm should be < 1e-9:', norm
        print 'terminating...'
        quit()

    print 'numerical gradient correct'

def compute_numerical_gradient(J, theta):
    '''
      theta:  a vector of parameters
      J:      a function that returns a tuple, (v, g), where v is the value of the
              cost function represented by J at theta, and grad is the gradient
    '''

    # numgrad[i] is the numerical approximation to the partial derivative of J with respect
    # to the (i+1)-th input argument, evaluated at theta
    numgrad = np.zeros(theta.shape)
    epsilon = 1e-4

    for i in xrange(len(numgrad)):
        theta[i] += epsilon
        numgrad[i] = J(theta)[0]

        theta[i] -= 2*epsilon
        numgrad[i] -= J(theta)[0]
        theta[i] += epsilon

        numgrad[i] /= 2*epsilon

    return numgrad
