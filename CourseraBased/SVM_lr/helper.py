from math import e
import numpy as np

def sigmoid(x):
    return 1/(1+e**-x)

def initialize_param(dim):
    w = np.zeros(shape=(dim, 1))
    b = 0
    return w, b

def propagate(w, b, X, Y):
    m = Y.shape[1]
    yhat = sigmoid(X.T@w+b)
    cost = (-1/m)*(np.dot(Y, np.log(yhat)) + np.dot((1-Y), np.log(1-yhat)))
    gradient = (((yhat - Y.T)*X.T[:]).sum(axis=0)).reshape(w.shape)/m
    gradient = {'dw': gradient, 'db':(yhat-Y.T).sum(axis=0)/m}
    return gradient, cost

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    grad, cost = propagate(w, b, X, Y)
    for i in range(num_iterations):
        grad, cost = propagate(w, b, X, Y)
        w = w - learning_rate*(grad['dw'])
        b -= learning_rate*grad['db']
        if i % 100 == 0 and print_cost:
            print(cost)
    param = {'w': w, 'b': b}
    return param, grad, cost

def predict(w, b, X):
    yhat = sigmoid(X.T@w+b)
    result = 1*(yhat>=0.5)
    return result

def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):
    w, b = initialize_param(X_train.shape[0])
    print(X_train.shape, w.shape, Y_train.shape, w.shape)
    parm, grad, cost = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    result_test = predict(parm['w'], parm['b'], X_test)
    result_train = predict(parm['w'], parm['b'], X_train)
    print("train accuracy: {} %".format(100 - np.mean(np.abs(result_train - Y_train.T)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(result_test - Y_test.T)) * 100))
    d = {"costs": cost,
         "Y_prediction_test": result_test,
         "Y_prediction_train": result_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d
