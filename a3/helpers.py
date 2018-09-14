import numpy as np
from math import e

def sigmoid(x):
    return 1/(1+np.exp(-x)), {'Z': x}


def relu(x):
    return np.maximum(x, 0), {'Z': x}


def initialize_parameters(a):
    x, h, y = a
    return {'W1': np.random.randn(h, x)*0.01,
            'b1': np.zeros((h, 1)),
            'W2': np.random.randn(y, h)*0.01,
            'b2': np.zeros((y, 1))}

def initialize_parameters_deep(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    dct = {}
    for i in range(len(layer_dims) - 1):
        dct['W{}'.format(i+1)] = np.random.randn(layer_dims[i+1], layer_dims[i])\
                                 /np.sqrt(layer_dims[i])
        dct['b{}'.format(i+1)] = np.zeros((layer_dims[i+1], 1))
    return dct


def linear_forward(A_prev, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass
    efficiently
    """
    return W@A_prev+b, {'A_prev': A_prev, 'W': W, 'b': b}


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    if activation == 'sigmoid':
        Z, linear = linear_forward(A_prev, W, b)
        A, activate = sigmoid(Z)
    elif activation == 'relu':
        Z, linear = linear_forward(A_prev, W, b)
        A, activate = relu(Z)
    return A, {'linear_cache': linear, 'activation_cache': activate}


def L_model_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_activation_forward() (there are L-1 of them, indexed from 0 to L-1)
    """
    l = int(0.5*len(parameters))
    A = X
    cache = []

    for i in range(1, l):
        A, temp = linear_activation_forward(A, parameters['W{}'.format(i)],
                                            parameters['b{}'.format(i)],
                                            'relu')

        cache.append({'linear_cache': temp['linear_cache'],
                      'activation_cache': temp['activation_cache']})

    AL, temp = linear_activation_forward(A, parameters['W{}'.format(l)],
                                         parameters['b{}'.format(l)],
                                         'sigmoid')
    cache.append({'linear_cache': temp['linear_cache'],
                  'activation_cache': temp['activation_cache']})

    return AL, cache


def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    return (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1 - AL).T))/(-Y.shape[1])


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache['A_prev'], cache['W'], cache['b']
    m = dZ.shape[1]
    dA_prev = W.T @ dZ
    dW = dZ @ A_prev.T / m
    db = np.sum(dZ) / m
    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    Z = cache['activation_cache']['Z']
    cache = cache['linear_cache']
    if activation == 'sigmoid':
        temp, useless = sigmoid(Z)
        dZ = np.multiply(dA, np.multiply(temp, 1 - temp))
        return linear_backward(dZ, cache)
    elif activation == 'relu':
        dZ = np.array(dA, copy=True)
        dZ[Z<0] = 0
        return linear_backward(dZ, cache)


def L_model_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group

    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
                the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])

    Returns:
    grads -- A dictionary with the gradients
    """
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    dAL = (np.divide(1 - Y, 1 - AL) - np.divide(Y, AL))
    current_cache = caches[-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, 'sigmoid')
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads['dA' + str(l + 1)], current_cache, 'relu')
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    k = len(parameters)/2
    for i in range(1, int(k) + 1):
        parameters['W' + str(i)] -= learning_rate*grads['dW' + str(i)]
        parameters['b' + str(i)] -= learning_rate*grads['db' + str(i)]
    return parameters

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2  # number of layers in the neural network
    p = np.zeros((1, m))

    # Forward propagation
    probas, caches = L_model_forward(X, parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0

    # print results
    # print ("predictions: " + str(p))
    # print ("true labels: " + str(y))
    print("Accuracy: " + str(np.sum((p == y) / m)))

    return p
