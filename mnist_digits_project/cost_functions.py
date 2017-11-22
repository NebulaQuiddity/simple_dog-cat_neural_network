# Written by Tanner Leonard
# # November 10, 2017

import numpy as np

# ignore RunTimeWarning that occurs during calculation
np.seterr('ignore')


def cross_entropy(activations, outputs):
    """
    returns the quadratic cost given the activations and the outputs
    :param activations: an array of activations
    :param outputs: an array of the correct labels (0 or 1)
    :return: a scalar representing the cost of the network
    """
    # CROSS ENTROPY: C = -1/n (sum [y(log( a)) + (1 - y)(log(1 - a))])

    # compute the sum
    element_1 = np.nan_to_num(np.multiply(outputs, np.log(activations)))
    element_2 = np.nan_to_num(np.multiply(1 - outputs, np.log(1 - activations)))
    ce_sum = np.add(element_1, element_2)

    # compute the average
    ce_sum = np.divide(ce_sum, activations.shape[0])

    return np.fabs(ce_sum)





