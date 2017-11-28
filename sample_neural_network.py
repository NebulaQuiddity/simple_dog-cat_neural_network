# Written by Tanner Leonard
# 11/26/2017 (yep I use mm/dd/yyyy sorry!)

# import dependencies
import feed_forward_neural_network as ffnn
import load_mnist_data as lmnist

# load in the data from files (NOTE: YOU WILL NEED TO CREATE THIS DATA FIRST (see README for instructions!))
trx, tr_y, tex, tey = lmnist.load_data_from_files(('X_training_data.npy', 'Y_training_data.npy',
                                                   'X_test_data.npy', 'Y_test_data.npy'))

# create an instance of a neural network
network = ffnn.FeedForwardNeuralNetwork(trx, tr_y, tex, tey, ([network shape]))

# if you have already created a model, you can load in the weights and biases
# by using line 18. Otherwise, use line 17 to train the network
network.train(10000, 1, 0, 'lr', 's', 'weights_file_name.npy', 'biases_file_name.npy', (True, 50), epsilon=0.01)
network.load_weights_and_biases('pre_trained_weights.npy', 'pre_trained_biases.npy')

# If you did not train, you will need to assign values to the activation functions and epsilon value
network.hidden_layer_activation_function = 'lr'
network.output_layer_activation_function = 's'
network.epsilon = 0.01

# now you can test the network visually with the training set or
# really test it with the test set
network.test_network_visually(network.input_X, network.input_Y)
correctly_classified = network.test_network()

# this just prints out the number classified out of the total number of images
print('Correctly classified in test set: ' + str(correctly_classified) + '/' + str(network.test_output_Y.shape[1]))
