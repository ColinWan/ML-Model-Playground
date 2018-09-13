import time
import numpy as np
import h5py
from lr_utils import load_dataset
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from helpers import *

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T
train_x = train_x_flatten/255.
test_x = test_x_flatten/255.

n_x = train_x_flatten.shape[0]
n_h = 7
n_y = 1
layers_dims = n_x, 7, n_y


def two_layer_model(X, Y, layers_dims, learning_rate=0.075, num_iterations=1000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations

    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    costs = []
    parameters = initialize_parameters_deep(layers_dims)
    AL, cache = L_model_forward(X, parameters)
    for i in range(num_iterations):
        grads = L_model_backward(AL, Y, cache)
        parameters = update_parameters(parameters, grads, learning_rate)
        AL, temp = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    return parameters, costs

model, cost = two_layer_model(train_x, train_y, layers_dims, print_cost=True)
predict_train, temp = L_model_forward(train_x, model)
print(predict_train, 'result')
# print(np.sum(predict_train-train_y)/train_y.shape[0])
