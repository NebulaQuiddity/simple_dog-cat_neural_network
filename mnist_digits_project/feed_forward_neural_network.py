# Written by Tanner Leonard
# x x, 2017

# import dependencies
import numpy as np
import activation_functions as af
import cost_functions as cf
import backprop_algorithms as ba

# TEMPORARY DEPENDENCIES
import load_mnist_data as lmnist
import time

# seed numpy for consistent results
np.random.seed(33234)


class FeedForwardNeuralNetwork:
    def __init__(self, training_data_input, training_data_output, shape):
        """

        :param training_data_input: this is the training data for the network. It should be a numpy array in which each
        column is a training input with n features, and if there are x training examples, this will be an n by x array
        :param training_data_output: the expected output of your neural network. This should be an x by 1 array in which
        each row is the output of the neural network.
        :param shape: this is the shape of your network. For example, if you had a network with three layers that had
        a neurons in the first layer (not including the input layer), b neurons in the second layer, and c neurons in
        the third layer, this parameter would be (a, b, c)
        """
        self.input_X = training_data_input
        self.output_Y = training_data_output
        assert self.input_X.shape[1] == self.output_Y.shape[1], 'the dimensions of the input ' \
                                                                + str(self.input_X.shape) + ' and output ' \
                                                                + str(self.output_Y.shape) + ' do not match'

        # the shape of the network with the input layer added
        shape.insert(0, self.input_X.shape[0])
        self.network_shape = shape

        # initialize the weights and biases randomly(these are essentially lists of arrays that correspond to each
        # layer)
        self.biases_B = [np.random.randn(x, 1) for x in self.network_shape[1:]]
        self.weights_W = [np.random.randn(shape[x], shape[x - 1])*0.1 for x in (range(len(shape))[1:])]

    def forward_propagate(self, input_batch_X, hidden_layer_activation_function, output_layer_activation_function,
                          *kwargs):
        """
        This function performs the forward propagation on a feed forward neural network. It returns an array, A, which
        contains the outputs for each of the inputs to the neural network
        :param input_batch_X: This is a matrix containing all of the inputs that you want to feed through the network.
        In the case that you are using a batch size of 1, this will be a vector of size n by 1 where n is the number of
        input features. In all other cases this should be a vector of size n by k where k is the batch size
        :param hidden_layer_activation_function: Only 2 supported as of now:
            's' for sigmoid
            'lr' for leaky relu
        :param output_layer_activation_function: Only 2 supported as of now:
            's' for sigmoid
            'lr' for leaky relu
        :param kwargs: if you are using leaky relu, please put the epsilon here (such as 0.01)
        :return: Z, A, AL(an array of the activations of the last layer of neurons for each training example)
        """

        self.hidden_layer_activation_function = hidden_layer_activation_function
        self.output_layer_activation_function = output_layer_activation_function
        if kwargs:
            self.epsilon = kwargs[0]
        else:
            self.epsilon = None

        # create a list to hold all of the z's for backpropogation
        Z = []

        #creat a list to hold all of the a's for backpropogation
        A = []

        # the first activations are just the inputs
        current_layer_activations = input_batch_X

        #iterate through every layer (except the input layer) and feed forward (activation(wx + b))
        for layer in range(len(self.network_shape[1:])):
            #compute z = wx + b
            z = np.dot(self.weights_W[layer], current_layer_activations) + self.biases_B[layer]
            #append z to use later
            Z.append(z)

            #apply the activation appropriate activation function to z to get a
            if layer != len(self.network_shape[1:]) - 1:
                current_layer_activations = af.activate(self.hidden_layer_activation_function, z, self.epsilon)

            else:
                current_layer_activations = af.activate(self.output_layer_activation_function, z, self.epsilon)

            # append a to use later
            A.append(current_layer_activations)

        Z = np.array(Z)
        A = np.array(A)

        self.z_activations = Z
        self.a_activations = A

    def cross_entropy_cost(self):
        """
        returns the cross entropy cost an an average over all of the activations
        :param activations: the activations of the final layer of the neural network
        :param y_outputs: the predicted outputs
        :return: a scalar that represents the overall cost
        """

        return cf.cross_entropy(self.a_activations[-1], self.output_Y)

    def backpropagate(self):
        """
        Propagates backward through the neural network layer by layer and calculates the partial derivatives the cost
        function with respect to every weight and bias
        :return: w_grad, b_grad where w_grad is the derivative of the weights of the network and z_grad is the
                 derivative of the biases of the network
        """

        # obtain the costs for each layer in the neural network
        layer_costs = self.layer_costs()

        # obtain the costs for the individual weights and biases in the network
        self.biases_B_cost = []
        self.weights_W_cost = []
        for x in range(len(self.biases_B)):
            self.biases_B_cost.append(layer_costs[x])
            self.weights_W_cost.append(np.dot(layer_costs[x], self.a_activations[x - 1].transpose()))

        # update the weights and the biases
        


    def layer_costs(self):
        """
        Calculates the cost for every layer in the network
        :param activation_function: the activation function for the last layer in the neural network
        :param L_activations: the activations of the last layer in the neural network
        :param outputs: the correct outputs
        :param z_activations: the weighted sums of the last layer in the neural network
        :param epsilon: the epsilon for leaky relu activation function
        :return: a list which contains arrays for every layer
        """
        # this list will hold the costs for all of the layers
        layer_costs = []

        # get the cost function for the final layer of the neural network
        output_layer_cost = ba.cross_entropy_prime(self.output_layer_activation_function, self.a_activations[-1],
                                                   self.output_Y, self.z_activations[-1], self.epsilon)
        layer_costs.append(output_layer_cost)

        # now, go backwards through the network and obtain the cost for all of the layers
        for layer in reversed(range(1, len(self.network_shape) - 1)):
            # multiply the transpose weight matrix of the next layer with the cost of the next layer
            element_1 = np.dot(self.weights_W[layer].transpose(), layer_costs[0])

            # return the element-wise product of element_1 and the derivative of the activation function
            layer_costs.insert(0, np.multiply(element_1, af.activate(self.hidden_layer_activation_function + 'p',
                                                                     self.z_activations[layer - 1], (self.epsilon))))

        return layer_costs




trx, tr_y, tex, tey = lmnist.load_data_from_files(('ffnn_numpy_data/X_train.npy', 'ffnn_numpy_data/Y_train.npy',
                                                   'ffnn_numpy_data/X_test.npy', 'ffnn_numpy_data/Y_test.npy'))
test_neural_network = FeedForwardNeuralNetwork(trx, tr_y, [12, 12, 11, 11, 10])
'''
print('tests: ')
print('\nnetwork shape: ' + str(test_neural_network.network_shape))
print('\nshape of bias vectors: ')
for x in test_neural_network.biases_B:
    print(x.shape)
print('\nbias vectors: ')
for x in test_neural_network.biases_B:
    print(x)
print('\nshape of weight arrays: ')
for x in test_neural_network.weights_W:
    print(x.shape)
print('\nweight arrays: ')
for x in test_neural_network.weights_W:
    print(x)
'''
time_start = time.process_time()
test_neural_network.forward_propagate(test_neural_network.input_X, 'lr', 's', 0.01)
c = test_neural_network.cross_entropy_cost()
test_neural_network.backpropagate()
time_end = time.process_time()
print(time_end - time_start)







