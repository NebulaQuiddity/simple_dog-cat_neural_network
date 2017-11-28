# Written by Tanner Leonard
# 11/25/2017 (yep I use mm/dd/yyyy sorry!)

# import dependencies
import numpy as np
import activation_functions as af
import cost_functions as cf
import backprop_algorithms as ba
from matplotlib import pyplot as plt

# seed numpy for consistent results
np.random.seed(29239)


class FeedForwardNeuralNetwork:
    def __init__(self, training_data_input, training_data_output, test_data_input, test_data_output, shape):
        """

        :param training_data_input: this is the training data for the network. It should be a numpy array in which each
        column is a training input with n features, and if there are x training examples, this will be an n by x array
        :param training_data_output: the expected output of your neural network. This should be an x by 1 array in which
        each row is the output of the neural network.
        :param test_data_input: this is the test data for the network. It should be a numpy array in which each
        column is a training input with n features, and if there are x training examples, this will be an n by x array
        :param test_data_output: the expected output of your neural network. This should be an x by 1 array in which
        each row is the output of the neural network.
        :param shape: this is the shape of your network. For example, if you had a network with three layers that had
        a neurons in the first layer (not including the input layer), b neurons in the second layer, and c neurons in
        the third layer, this parameter would be (a, b, c)
        """
        self.input_X = training_data_input
        self.output_Y = training_data_output

        self.test_input_X = test_data_input
        self.test_output_Y = test_data_output

        self.epsilon = None
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

    def train(self, batch_size, iterations, learning_rate, hlactivation_function, olactivation_function, weights_file_name,
              biases_file_name, print_cost, epsilon=None):
        """
        trains the neural network
        :param batch_size: the batch size for training, will be randomly selected
        :param iterations: number of total training iterations
        :param hlactivation_function: the activation function for the hidden layer ('s' for sigmoid,
                                      lr' for leaky relu)
        :param olactivation_function: the activation function for the output layer ('s' for sigmoid,
                                      lr' for leaky relu)
        :param weights_file_name: the name of the file that you want the program to save the weights to (should be .npy)
        :param biases_file_name: the name of the file that you want the program to save the biases to (should be .npy)
        :param print_cost: a tuple that contains 2 elements (True or False, Number of iterations to print the cost)
        :param epsilon: the epsilon needed for the leaky relu activation function
        :return: None
        """

        #create a list to store the costs of the neural network for each iteration
        iteration_costs = []

        for iteration in range(iterations):
            #shuffle the input array and choose the first x elements for the training batch
            p = np.random.permutation(len(self.input_X.transpose()))
            a = self.input_X.transpose()[p]
            b = self.output_Y.transpose()[p]
            self.training_batch_X = a[:batch_size].transpose()
            self.training_batch_Y = b[:batch_size].transpose()

            #forward propagate through the neural network
            self.forward_propagate(self.training_batch_X, hlactivation_function, olactivation_function, epsilon)

            #print the cost if wanted
            if print_cost:
                if iteration % print_cost[1] == 0:
                    iter_cost = np.sum(self.cross_entropy_cost(self.training_batch_Y))/self.training_batch_X.shape[1]
                    print('Iteration: ' + str(iteration) + ', Cost: ' + str(iter_cost))

            #backpropogate through the network and update the weights and biases
            self.backpropagate(self.training_batch_Y, learning_rate)

        # save the arrays to files
        np.save(weights_file_name, np.array(self.weights_W))
        np.save(biases_file_name, np.array(self.biases_B))

    def load_weights_and_biases(self, weights_file_name, biases_file_name):
        """
        allow you to load in previous weight and bias arrays from previous training
        NOTE THAT IF YOU USE THE LEAKY RELU ACTIVATION FUNCTION YOU WILL ALSO HAVE TO SET AN EPSILON!!!
        :param weights_file_name: the file name of the weights of the neural network
        :param biases_file_name: the file name of the biases of the neural network
        :return: none
        """
        w = np.load(weights_file_name)
        b = np.load(biases_file_name)

        # load in the previously trained weights and biases
        self.weights_W = []
        self.biases_B = []
        for x in range(len(self.network_shape) - 1):
            self.weights_W.append(w[x])
            self.biases_B.append(b[x])


    def test_network(self):
        """
        iterate through the network with the whole test set of images
        :return: a where a is the number of images correctly classified by the neural network
        """
        # forward propagate through the network
        self.forward_propagate(self.test_input_X, self.hidden_layer_activation_function,
                               self.output_layer_activation_function, self.epsilon)

        # compare the results to the correct outputs
        classified = 0

        for x in range(self.a_activations[-1].shape[1]):
            if int(list(self.a_activations[-1].transpose()[x]).index(np.amax(self.a_activations[-1].transpose()[x])))\
                == int(list(self.test_output_Y.transpose()[x]).index(np.amax(self.test_output_Y.transpose()[x]))):
                classified += 1

        return classified


    def test_network_visually(self, images):
        self.compare_image_to_class(images)

    def compare_image_to_class(self, image_array):
        """
        allows you to visually compare what the neural network predicts to the image
        :param image_array: the input array that contains all of the image data
        :return: nothing
        """
        images = []
        img_array = image_array.transpose()

        for x in img_array:
            images.append(x.reshape(int(np.sqrt(x.shape[0])), int(np.sqrt(x.shape[0]))))

        active = True
        while active:
            active = self.show_images(images)

    def show_images(self, images):
        """
        shows images using matplotlib and prints the correct classification
        :param images: the images that you want to view
        :return: nothing
        """

        image_number = input('What image would you like to view? Type \\quit to quit. ')

        if image_number == '\\quit':
            return False


        image_number = int(image_number)

        image = np.array(images[image_number])
        a = self.input_X.transpose()[image_number].transpose().reshape(self.input_X.transpose()[image_number].transpose().shape[0], 1)
        self.forward_propagate(a, self.hidden_layer_activation_function, self.output_layer_activation_function,
                               self.epsilon)
        print('classified as: ' + str(list(self.a_activations[-1]).index(np.amax(self.a_activations[-1]))))

        plt.imshow(image, cmap='Greys'), plt.axis('off')
        plt.show()
        plt.close('all')

        return True

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

    def cross_entropy_cost(self, outputs):
        """
        returns the cross entropy cost as an average over all of the activations
        :param activations: the activations of the final layer of the neural network
        :param y_outputs: the predicted outputs
        :return: a scalar that represents the overall cost
        """

        return cf.cross_entropy(self.a_activations[-1], outputs)

    def backpropagate(self, outputs, learning_rate):
        """
        Propagates backward through the neural network layer by layer and calculates the partial derivatives the cost
        function with respect to every weight and bias
        :return: w_grad, b_grad where w_grad is the derivative of the weights of the network and z_grad is the
                 derivative of the biases of the network
        """

        # obtain the costs for each layer in the neural network
        nn_layer_costs = self.layer_costs(outputs)

        # obtain the costs for the individual weights and biases in the network
        self.biases_B_cost = []
        self.weights_W_cost = []
        for x in range(len(self.biases_B)):
            self.biases_B_cost.append(nn_layer_costs[x])
            if x - 1 == -1:
                self.weights_W_cost.append(np.dot(nn_layer_costs[x], self.training_batch_X.transpose()))
            else:
                self.weights_W_cost.append(np.dot(nn_layer_costs[x], self.a_activations[x - 1].transpose()))


        # update the weights and the biases
        for x in range(len(self.weights_W)):
            self.weights_W[x] = np.subtract(self.weights_W[x], learning_rate * np.multiply((1/self.a_activations[-1].shape[1]),
                                                                      self.weights_W_cost[x]))
            self.biases_B[x] = np.subtract(self.biases_B[x], learning_rate * np.multiply((1/self.a_activations[-1].shape[1]),
                                                                    np.sum(self.biases_B_cost[x], 1, keepdims=True)))

    def layer_costs(self, outputs):
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
                                                   outputs, self.z_activations[-1], self.epsilon)
        layer_costs.append(output_layer_cost)

        # now, go backwards through the network and obtain the cost for all of the layers
        for layer in reversed(range(1, len(self.network_shape) - 1)):
            # multiply the transpose weight matrix of the next layer with the cost of the next layer
            element_1 = np.dot(self.weights_W[layer].transpose(), layer_costs[0])

            # return the element-wise product of element_1 and the derivative of the activation function
            layer_costs.insert(0, np.multiply(element_1, af.activate(self.hidden_layer_activation_function + 'p',
                                                                     self.z_activations[layer - 1], (self.epsilon))))

        return layer_costs
