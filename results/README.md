# Results

Here are some results for a 3-layer autoencoder. The first and last layers
have 81 units, each representing a pixel in a 9x9 image. The second -- hidden --
layer has 25 units. Every unit uses a sigmoid activation function.

The training set is 50 randomly generated 9x9 rectangles (each pixel taking a value
in [0,1]). The cost function is optimized using an open-source implementation of
[L-BGFS-B](http://en.wikipedia.org/wiki/Limited-memory_BFGS).

The training set is 20 random 9x9 rectangles, shown below.
![training set](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/xs.png)

## 1. Simple autoencoder
The cost funciton is the average [L^2 norm](http://en.wikipedia.org/wiki/Euclidean_distance) between the output and the input images.

The learned weights for the hidden layer:
![weights 1](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/hidden_unit_weights_1.png)

The output for the test data:
![x hats 1](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/x_hats_1.png)

Average error: 2.68, about ?% accurate. Trains in 51 iterations of the optimization
algorithm.

## 2. Adding weight decay regularization
Now, the magnitude of the weight parameters is
included in the cost function. This change, as a type of regularization, prevents
overfitting. The sum of squares of the weight parameters in the network decreases
to 0.6% of its previous, non-regularized size.

![weights 2](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/hidden_unit_weights_2.png)

![x hats 2](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/x_hats_2.png)

Average error: 1.64, about % accurate. Trains in 779 iterations.

## 3. Adding sparsity
We want our network to learn structured features of our data. Thus, we enforce that the
hidden units have sparse activations for a given input. That is, we want a low number of
activations to capture the higher dimensionality features our data exhibits.

The cost function now penalizes when the average activation for a hidden unit
deviates from some sparsity term we define (in the code, we use 1% sparsity, and use
[Kullback-Leibler divergence](wiki link)).

![weights 3](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/hidden_unit_weights_3.png)

![x hats 3](https://github.com/FluxLemur/UnsupervisedLearning/blob/master/results/x_hats_3.png)

Average error: 3.24, about ?% accurate. Trains in ? iterations.
