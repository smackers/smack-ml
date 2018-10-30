#!/usr/bin/env python
import os, pstats, sys
import random
import numpy as np
import pandas as pd
#import mnist_loader
import cPickle as pickle
import matplotlib.pyplot as plt

class Network(object):
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.j = 0

		'''----------------------------if n=1, initialize bias & weight randomly-------------------------'''
		if n == 1:
			print "Initializing through randomizer: Creating new random bias and weights"
			self.biases = [np.random.randn(y,1) for y in sizes[1:]]
			self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
		else:
			 if os.stat("Bias_ann.p").st_size >= 9000 or os.stat("Weights_ann.p").st_size == 0:
                        	print "Max file size reached or no previous log exists.."
				print "Clearing the saved Bias and Weights..creating new random bias and weights"
	                        open("Bias_ann.p","w").close()
        	                open("Weights_ann.p","w").close()
                	        self.biases = [np.random.randn(y,1) for y in sizes[1:]]
                     		self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1],sizes[1:])]
  			 else:
 				print "Unpickle: Loading previously stored biases and weights from respective files.."
				self.biases = pickle.load(open("Bias_ann.p","rb"))
				self.weights = pickle.load(open("Weights_ann.p","rb"))

		print "Starting the algorithm"

	def feedforward(self, a):
		"""Return the output of the network if ``a`` is input."""
		for b, w in zip(self.biases, self.weights):
		    a = sigmoid(np.dot(w, a)+b)
		return a

	def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
		"""Train the neural network using mini-batch stochastic
		gradient descent.  The ``training_data`` is a list of tuples
		``(x, y)`` representing the training inputs and the desired
		outputs.  The other non-optional parameters are
		self-explanatory.  If ``test_data`` is provided then the
		network will be evaluated against the test data after each
		epoch, and partial progress printed out.  This is useful for
		tracking progress, but slows things down substantially."""
		if test_data: n_test = len(test_data)
		n = len(training_data)
		for j in xrange(epochs):
		    self.i = j
		    random.shuffle(training_data)
		    mini_batches = [
		        training_data[k:k+mini_batch_size]
		        for k in xrange(0, n, mini_batch_size)]
		    for mini_batch in mini_batches:
		        self.update_mini_batch(mini_batch, eta)
		    if test_data:
		        print "Epoch {0}: {1}".format(
		            j, float(self.evaluate(test_data)*100.0 /n_test))
		    else:
		        print "Epoch {0} complete".format(j)

	def update_mini_batch(self, mini_batch, eta):
		"""Update the network's weights and biases by applying
		gradient descent using backpropagation to a single mini batch.
		The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
		is the learning rate."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		for x, y in mini_batch:
		    delta_nabla_b, delta_nabla_w = self.backprop(x, y)
		    nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
		    nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
		self.weights = [w-(eta/len(mini_batch))*nw
		                for w, nw in zip(self.weights, nabla_w)]
		self.biases = [b-(eta/len(mini_batch))*nb
		               for b, nb in zip(self.biases, nabla_b)]

		'''dump the bias and weights in a file after every 5 epochs
			(helps in restarting even if terminated in between)'''
		if (self.i+1) % 200 == 0: #(i+1) since i starts from 0 so the last iteration will be skipped otherwise
			'''Dumping the bias and weights into a file for later use'''
			print "cPickle: Updating the bias and weight files"
			pickle.dump(self.biases,open("Bias_ann.p","wb"))
			pickle.dump(self.weights,open("Weights_ann.p","wb"))

	def backprop(self, x, y):
		"""Return a tuple ``(nabla_b, nabla_w)`` representing the
		gradient for the cost function C_x.  ``nabla_b`` and
		``nabla_w`` are layer-by-layer lists of numpy arrays, similar
		to ``self.biases`` and ``self.weights``."""
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]
		# feedforward
		activation = x
		activations = [x] # list to store all the activations, layer by layer
		zs = [] # list to store all the z vectors, layer by layer
		for b, w in zip(self.biases, self.weights):
		    z = np.dot(w, activation)+b
		    zs.append(z)
		    activation = sigmoid(z)
		    activations.append(activation)
		# backward pass
		delta = self.cost_derivative(activations[-1], y) * \
		    sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta, activations[-2].transpose())
		# Note that the variable l in the loop below is used a little
		# differently to the notation in Chapter 2 of the book.  Here,
		# l = 1 means the last layer of neurons, l = 2 is the
		# second-last layer, and so on.  It's a renumbering of the
		# scheme in the book, used here to take advantage of the fact
		# that Python can use negative indices in lists.
		for l in xrange(2, self.num_layers):
		    z = zs[-l]
		    sp = sigmoid_prime(z)
		    delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
		    nabla_b[-l] = delta
		    nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
		return (nabla_b, nabla_w)

	def evaluate(self, test_data):
		"""Return the number of test inputs for which the neural
		network outputs the correct result. Note that the neural
		network's output is assumed to be the index of whichever
		neuron in the final layer has the highest activation."""
		#test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
		test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x,y) in test_data]
		return sum(int(x == y) for (x, y) in test_results)

	def cost_derivative(self, output_activations, y):
		"""Return the vector of partial derivatives \partial C_x /
		\partial a for the output activations."""
		return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def formatting(X, Y):
	labels = []; Xf = [];

	for i in range(len(X)):
	    tmp = [[i] for i in X[i]]
	    Xf.append(tmp)

	for i in range(len(Y)):
	    tmp = [[0]]*11
	    tmp[Y[i]] = [1]
	    labels.append(tmp)

	input_data = []
	for f, l in zip(Xf, labels):
	    input_data.append((np.array(f), np.array(l)))

	return input_data


if __name__ == '__main__':
    #load data
    dataset = pd.read_csv('trainXY.csv',sep=' ')
    X = dataset.iloc[:,1:-1].values
    y = dataset.iloc[:,-1].values


    # Splitting the dataset into the Training set and Test set
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

    # feature scaling
    from sklearn.preprocessing import StandardScaler
    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    #print X_train[0]
    training_data = formatting(X_train, y_train)
    test_data = formatting(X_test, y_test)

    n = input("Enter '1' to restart (any other key to resume): ")
    #initializing network (# of neurons in each layer size(net) = #of layers)
    net = Network([33,25,11])
    net.SGD(training_data,1000,20,0.00095,test_data)

    #plotting SVD and saving the figures
    U, S, V = np.linalg.svd(net.weights[0],full_matrices = True)
    plt.plot(S)
    plt.savefig("svd.png")
