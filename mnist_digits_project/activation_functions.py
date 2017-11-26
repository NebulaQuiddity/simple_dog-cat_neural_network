# Written by Tanner Leonard
# 11/25/2017 (yep I use mm/dd/yyyy sorry!)

# import dependencies
import numpy as np


def sigmoid(input_array):
    """
    returns an array with the sigmoid function applied to each element
    :param input_array: a numpy array
    :return: input_array with the sigmoid function applied
    """

    s = np.divide([1], (1 + np.exp(-1 * input_array)))
    return s


def sigmoid_prime(input_array):
    """
    returns an array with the derivative of the sigmoid function applied to each element
    :param input_array: a numpy array
    :return: input_array with derivative applied
    """

    ds = np.multiply(sigmoid(input_array), (1 - sigmoid(input_array)))
    return ds


def leaky_relu(input_array, epsilon):
    """
    returns an array with the leaky relu activation function applied
    :param input_array: a numpy array
    :param epsilon: a small number
    :return: input_array with leaky relu activatioin applied
    """

    lr = np.piecewise(input_array.astype(float), [input_array < 0, input_array >= 0],
                      [lambda x: np.multiply(x, epsilon), lambda x: x])
    return lr


def leaky_relu_prime(input_array, epsilon):
    """
    returns an array with the derivative of the leaky relu function applied to each element
    :param input_array: a numpy array
    :param epsilon: a small number
    :return: a numpy array
    """

    dlr = np.piecewise(input_array.astype(float), [input_array <= 0, input_array > 0],
                       [lambda x: epsilon, lambda x: 1])
    return dlr


def activate(activation_function, input_array, *kwargs):
    """
    returns any activation function (just for convenience
    :param activation_function: the activation function that you want to use
    :param input_array: the array that you want to apply the activation function to
    :return: the output of the activation function
    """

    if activation_function == 's':
        return sigmoid(input_array)

    if activation_function == 'sp':
        return sigmoid_prime(input_array)

    if activation_function == 'lr':
        return leaky_relu(input_array, kwargs[0])

    if activation_function == 'lrp':
        return leaky_relu_prime(input_array, kwargs[0])
