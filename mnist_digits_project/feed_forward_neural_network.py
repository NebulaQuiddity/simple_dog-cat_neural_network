# Written by Tanner Leonard
# x x, 2017

# import dependencies
import numpy as np
import activation_functions as af
import cost_functions as cf

# TEMPORARY DEPENDENCIES
import load_mnist_data as lmnist
import time

# seed numpy for consistent results
np.random.seed(23492)


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
        pass


''
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
test_neural_network.forward_propagate(test_neural_network.input_X, 'lr', 's', 0.01)
c = test_neural_network.cross_entropy_cost()


