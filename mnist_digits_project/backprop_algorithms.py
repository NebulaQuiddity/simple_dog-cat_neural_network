import numpy as np
import activation_functions as af

def cross_entropy_prime(activation_function, L_activations, outputs, z_activations, epsilon=None):
    """
    returns the partial derivative of the cost function with respect to the activations in the last layer
    :param activation_function: the activation function for the last layer in the neural network
    :param L_activations: the activations of the last layer of the neural network
    :param outputs: the correct outputs
    :param z_activations: the weighted sums of the last layer in the neural network
    :param epsilon: the epsilon for leaky relu activation function
    :return: a numpy array containing the partial derivatives
    """

    # compute -y/a + (1 - y)/(1 - a)
    dc_dal = np.add(np.multiply(-1, np.divide(outputs, L_activations)),
                    np.divide(np.subtract(1, np.multiply(-1, outputs)),
                              np.subtract(1, np.multiply(-1, L_activations))))

    # return the element-wise product of dc_dal with the derivative of the activation function
    if activation_function == 'lr':
        return np.multiply(af.activate('lrp', z_activations, (epsilon)), dc_dal)

    elif activation_function == 's':
        return np.multiply(af.activate('sp', z_activations), dc_dal)

def layer_costs(activation_function, L_activations, outputs, z_activations, epsilon=None):
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
    output_layer_cost = cross_entropy_prime(activation_function, L_activations, outputs,
                                               z_activations, epsilon)
    layer_costs.append(output_layer_cost)

    # now, go backwards through the network and obtain the cost for all of the layers


