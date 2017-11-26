# Written by Tanner Leonard
# 11/26/2017 (yep I use mm/dd/yyyy sorry!)

# import dependencies
import feed_forward_neural_network as ffnn
import load_mnist_data as lmnist

# load in the data from files (NOTE, YOU WILL NEED TO CREATE THIS DATA FIRST!)
trx, tr_y, tex, tey = lmnist.load_data_from_files(('ffnn_numpy_data/X_train.npy', 'ffnn_numpy_data/Y_train.npy',
                                                   'ffnn_numpy_data/X_test.npy', 'ffnn_numpy_data/Y_test.npy'))

# create an instance of a neural network
network = ffnn.FeedForwardNeuralNetwork(trx, tr_y, tex, tey, [70, 30, 12, 10])

# if you have already created a model, you can load in the weights and biases
# by using line 18. Otherwise, use line 17 to train the network
# network.train(10000, 1, 0, 'lr', 's', 'weights.npy', 'biases.npy', (True, 50), epsilon=0.01)
network.load_weights_and_biases('weights_new.npy', 'biases_new.npy')

# also, I need to specify the hidden layer and output layer activation functions
network.hidden_layer_activation_function = 'lr'
network.output_layer_activation_function = 's'
network.epsilon = 0.01

# now you can test the network visually with the training set or
# really test it with the test set
network.test_network_visually(network.input_X)
correctly_classified = network.test_network()
print(str(correctly_classified) + '/' + str(network.test_output_Y.shape[1]))
