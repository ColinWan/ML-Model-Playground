import numpy as np
from math import e

def sigmoid(x):
    return 1/(1+e**-x)

def net_size(X, Y, h=4):
    return X.shape[0], h, Y.shape[0]

def initial_param(x, h, y):
    return {'W1':np.random.randn(h, x)*0.01,
            'b1':np.zeros((h, 1)),
            'W2':np.random.randn(y, h)*0.01,
            'b2':np.zeros((y, 1))}

def forward_prop(X, param):
    W1 = param['W1']
    b1 = param['b1']
    W2 = param['W2']
    b2 = param['b2']

    Z1= W1 @ X + b1
    A1= np.tanh(Z1)
    Z2= W2 @ A1 + b2
    A2= sigmoid(Z2)
    return A2, {'Z1':Z1, 'A1':A1, 'Z2':Z2, 'A2':A2}

def cost(A2, Y):
    m = Y.shape[1]
    cost = (-1/m)*(Y@np.log(A2).T + ((1-Y)@np.log(1-A2).T))
    return cost

def back_prop(param, cache, X, Y):
    m = Y.shape[1]
    A2, cache = forward_prop(X, param)
    dz2 = A2 - Y
    dw2 = (1/m)*(dz2@cache['A1'].T)
    db2 = (1/m)*np.sum(dz2, axis=1)
    # db2 = db2.reshape(db2.shape[0], 1)
    dz1 = param['W2'].T*dz2*(1-np.power(cache['A1'], 2))
    dw1 = (1/m)*(dz1@X.T)
    db1 = (1/m)*np.sum(dz1, axis=1)
    db1 = db1.reshape(db1.shape[0], 1)
    return {'dW2': dw2,
            'dW1': dw1,
            'db1': db1,
            'db2': db2}

def update(param, grads, rate=1.2):
    for item in param:
        grad = 'd' + item
        param[item] = param[item] - rate*grads[grad]
    return param

def predict(parameters, X):
    A2, cache = forward_prop(X, parameters)
    predictions = np.round(A2)
    return predictions
