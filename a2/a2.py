# Package imports
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
from helper import net_size, initial_param, forward_prop, cost, back_prop, update, predict
np.random.seed(3)

# Load Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}
dataset = 'gaussian_quantiles'
# X, Y = load_planar_dataset()
X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# Model
def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    a, b, c = net_size(X, Y)
    b = n_h
    param = initial_param(a, b, c)
    for i in range(0, num_iterations):
        A2, cache = forward_prop(X, param)
        c = cost(A2, Y)
        grads = back_prop(param, cache, X, Y)
        param = update(param, grads)

        if print_cost and i % 1000 == 0:
            print('Cost is ' + str(c))

    return param

# Visual
parameters = nn_model(X, Y, 4, num_iterations = 50000, print_cost=True)
# Plot the decision boundary
plot_decision_boundary(lambda x: predict(parameters, x.T).ravel(), X, Y.ravel())
predictions = predict(parameters, X)
accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
print ("Accuracy for {} hidden units: {} %".format(4, accuracy))
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()

# # Net Size
# plt.figure(figsize=(16, 32))
# hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
# for i, n_h in enumerate(hidden_layer_sizes):
#     plt.subplot(5, 2, i+1)
#     plt.title('Hidden Layer of size %d' % n_h)
#     parameters = nn_model(X, Y, n_h, num_iterations = 5000)
#     plot_decision_boundary(lambda x: predict(parameters, x.T).ravel(), X, Y.ravel())
#     predictions = predict(parameters, X)
#     accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
#     print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
plt.show()
