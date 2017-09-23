'''
Written by Tanner Leonard
September 22, 2017
'''


#import dependencies
import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import scipy
import scipy.misc


def image_to_array(image_samples, name_beginning, name_end, crop=True, crop_shape='square', average_size=None):
    '''
    turns a set of images into a numpy array of shape (image_samples, dimension_x, dimension_y, 3(color channels))
    Please note that in order for this function to work, you must have all of your images in a single folder with a
    common name, e.g. 'train/cat.x.jpg' where x is the image number. This function will crop the images if wanted, and
    will also resize them to the average size

    :param test_data_samples: the number of images in your dataset with an orderly name
    :param name_beginning: the string at the beginning of all of your images' names. e.g. 'train/cat.'
    :param name_end: the string at the end of all of your images' names. e.g. '.jpg'
    :param crop: if you would like to crop your images, leave this as true. Currently, the program only crops your
    images into a square
    :param crop_shape: the shape you would like to crop your images into. However, currently the only shape supported is
    a 'square'
    :param average_size: this parameter should be set to None the first time you run this, however if you are applying
    this function to multiple datasets (e.g. a cat picture dataset and a dog picture dataset) you can pass in the
    average computed the first time to ensure that all of your images will be shaped to the same size
    :return: a numpy array containing your images of the shape (image_samples, dimension_x, dimension_y, 3(color channels)
    '''
    #this controls whether the average size needs to be updated
    update = False

    # a empty list to temporarily store the dataset images
    formatted_images = []

    # an empty variable used to determine the average size of your training images
    if average_size == None:
        average_size = 0
        update = True

    #iterate through the training images, convert them into arrays, and append them to the list of images
    for image in range(image_samples):
        #load an image as an array
        image_bgr = cv2.imread(name_beginning + str(image) + name_end, cv2.IMREAD_COLOR)
        #convert it to rgb
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.squeeze()

        #basically turn the image into a centered square
        if crop:
            #return the cropped image
            cropped_image = crop_image(crop_shape, image_rgb, (int(image_rgb.shape[0]), int(image_rgb.shape[1])))
            #append this image to the image list
            formatted_images.append(cropped_image)
            #add the average image size
            if update:
                average_size += (cropped_image.shape[0] + cropped_image.shape[1])/2.0

    #calculate the average image size
    if update:
        average_size = int(math.floor(average_size / image_samples))
    #turn the list into an array
    formatted_images = np.asarray(formatted_images)

    #create a new list for the cropped images
    cropped_images = []

    #resize the images
    for image in formatted_images:
        #resize the images
        new_image = scipy.misc.imresize(image, (average_size, average_size))

        #update the image
        cropped_images.append(new_image)

    #make sure formatted_images is an array
    cropped_images = np.asarray(cropped_images)

    #return the cropped images
    return cropped_images, average_size



def crop_image(shape, image, dimensions):
    '''
    This function crops an image (an array of values)

    :param shape: the shape of the crop (can only be square right now)
    :param dimensions: the dimensions of the image
    :param image: the image you want to crop
    :return: the cropped image
    '''

    #crop the image into a square
    if shape == 'square':
        #if the first dimension is larger than the second
        if dimensions[0] > dimensions[1]:
            #we need to make a square the size of the smaller dimensions
            square_size = dimensions[1]
            difference = dimensions[0] - square_size

            # the dimensions are even
            if (difference) % 2 == 0:
                cropped_image = image[int(difference / 2): int(dimensions[0] - (difference / 2))]
            # the dimensions are odd
            elif (difference) % 2 != 0:
                difference1 = (difference + 1) / 2
                difference2 = (difference - 1) / 2
                cropped_image = image[int(difference1): int(dimensions[0] - difference2)]
            new_image_dims = cropped_image.shape
            assert new_image_dims[0] == new_image_dims[1], 'cropping failed'

            #return cropped image
            return cropped_image

        #if the second dimension is larger than the second
        elif dimensions[0] < dimensions[1]:
            #we need to make a square the size of the smaller dimension
            square_size = dimensions[0]
            difference = dimensions[1] - square_size
            #the dimensions are even
            if (difference) % 2 == 0:
                cropped_image = image[:, int(difference/2): int(dimensions[1] - (difference / 2))]
            #the dimensions are odd
            elif (difference) %2 != 0:
                difference1 = (difference + 1)/2
                difference2 = (difference - 1)/2
                cropped_image = image[:, int(difference1): int(dimensions[1] - difference2)]

            #check to see if the crop was successful
            new_image_dims = cropped_image.shape
            assert new_image_dims[0] == new_image_dims[1], 'cropping failed'

            #return cropped image
            return cropped_image

    assert image.shape[0] == image.shape[1], 'cropping failed'
    return image

def view_cat_and_dog_training_data(cat_array, dog_array):
    '''
    allows you to interact with and view the cat and dog training data
    :param cat_array: An array of the cat images
    :param dog_array: An array of the dog images
    :return: None
    '''
    # ask the user what they want to do
    data_type = input(
        'Would you like to take a look at dog images (\'d\') or cat images (\'c\')? Type \\quit to quit: ')

    # allow the user to view the dog images
    if data_type == 'd':
        image_number = input('Please enter the image number you would like to view: ')

        plt.imshow(dog_array[int(image_number)]), plt.axis('off')
        plt.show()
        return True

    # allow the user to view the cat images
    if data_type == 'c':
        image_number = input('Please enter the image number you would like to view: ')

        plt.imshow(cat_array[int(image_number)]), plt.axis('off')
        plt.show()
        return True

    # allow the user to quit
    if data_type == '\\quit':
        return False

def create_cat_and_dog_training_data(interact=False, viewing_set=True):
    '''
    this function creates correctly formatted arrays for the cat/dog image library
    NOTE: make sure that the training images are named correctly and in a folder called 'train'
    :param interact: this will let you interact with the data, letting you view some of the images and see the
    dataset's shape
    :param viewing_set: set to True if you would also like an array that will enable you to view the data.
    :return: the training data arrays, along with the viewing set arrays if wanted
    '''
    if interact == True:
        print('Loading training data...')

    #load the dog and cat training data
    cat_training_images, avg_size = image_to_array(12500, 'train/cat.', '.jpg')
    dog_training_images, average = image_to_array(12500, 'train/dog.', '.jpg', crop=True, crop_shape='square', average_size=int(avg_size))

    if interact == True:
        print('Done!')
        print('\n\n\n')

    if interact == True:
        # alert the user that the training data is done loading
        print('Done!')
        print('\n\n\n')

        #start a loop where the user can access and view their images
        active = True
        while (active):
            active = view_cat_and_dog_training_data(cat_training_images, dog_training_images)

    # create 'network sets' of the images that are of the appropriate shape to be used as inputs in a neural network
    cat_network_set = cat_training_images.reshape(cat_training_images.shape[0], -1).T
    dog_network_set = dog_training_images.reshape(dog_training_images.shape[0], -1).T

    if interact==True:
        # tell the user the shape of their arrays
        print('Your cat and dog viewing sets are of shape ' + str(cat_training_images.shape) + ' and ' + \
              str(dog_training_images.shape) + ', and your cat and dog network sets are of shape ' + \
              str(cat_network_set.shape) + ' and ' + str(dog_network_set.shape) + '.')

    if viewing_set:
        # return the network sets and the viewing sets
        return cat_training_images, dog_training_images, cat_network_set, dog_network_set

    else:
        # return just the network sets
        return cat_network_set, dog_network_set

def format_image_arrays(X_filename, Y_filename, viewing_set=False, cat_filename=None, dog_filename=None):
    '''
    this function will complete the final step, loading the cat and dog data into X and Y arrays, along with loading
    the viewing data into files if specified
    :param X_filename: the name of the file in which the X array will be saved
    :param Y_filename: the name of the file in which the Y array will be saved
    :param viewing_set: set this to true in order to also save the viewing sets
    :param cat_filename: the name of the file in which the cat viewing array will be saved
    :param dog_filename: the name of the file in which the dog viewing array will be saved
    :return: None
    '''

    #load in the arrays of data
    cat_viewing_set, dog_viewing_set, cat_network_set, dog_network_set = create_cat_and_dog_training_data(
        interact=False, viewing_set=True)

    #create a dictionary to store the X and Y arrays
    cat_and_dog_dict = {}

    #I don't remember why I did this but I guess the data needed to be transposed
    cat_network_set = cat_network_set.T
    dog_network_set = dog_network_set.T

    #create lists to hold the X and Y data
    X = []
    Y = []

    #append the values to the X and Y lists
    for x in range(len(cat_network_set)):
        X.append(cat_network_set[x])
        Y.append(1)
        X.append(dog_network_set[x])
        Y.append(0)

    #convert X and Y to arrays, and reshape them
    X = np.asarray(X)
    Y = np.asarray(Y)
    X = X.reshape(X.shape[0], -2).T
    Y = Y.reshape(Y.shape[0], -2).T

    # save the X and Y arrays
    np.save(X_filename, X)
    np.save(Y_filename, Y)

    if viewing_set:
        # squeeze and save the viewing arrays
        cat_viewing_set = cat_viewing_set.squeeze()
        dog_viewing_set = dog_viewing_set.squeeze()

        np.save(cat_filename, cat_viewing_set)
        np.save(dog_filename, dog_viewing_set)


def setup_kaggle_dog_and_cat_dataset(X_filename, Y_filename, view=False,
                                     cat_viewing_filename=None, dog_viewing_filename=None):
    '''
    Run this function if you need to set up all of the dog and cat data on your computer. After you have run
    this, just run 'access_dog_and_cat_data' to load in your data.
    :param X_filename: the filename that the X data will be saved to
    :param Y_filename: the filename that the Y data will be saved to
    :param view: set this to true in order to also save a copy of the viewing set
    :param cat_viewing_filename: the filename that the cat viewing data will be saved to
    :param dog_viewing_filename: the filename that the dog viewing data will be saved to
    :return: nothing
    '''
    # create and save the image data
    format_image_arrays(X_filename, Y_filename, viewing_set=view, cat_filename=cat_viewing_filename,
                        dog_filename=dog_viewing_filename)


def load_dog_and_cat_training_data(X_filename, Y_filename, viewing=False, cat_viewing_filename=None,
                                   dog_viewing_filename=None):
    '''
    Run this function to load already created data into your script
    :param X_filename: the location of the X data
    :param Y_filename: the location of the Y data
    :param viewing: set this to True to also load the viewing data
    :param cat_viewing_filename: the location of the cat viewing data
    :param dog_viewing_filename: the location of the dog viewing data
    :return: the X and Y arrays, along with the viewing arrays if specified
    '''

    #load the X and Y arrays
    X = np.load(X_filename)
    Y = np.load(Y_filename)
    X = X.squeeze()
    Y = Y.squeeze()

    if viewing == False:
        return X, Y
    elif viewing == True:
        #load the cat and dog viewing arrays
        cat_view = np.load(cat_viewing_filename)
        dog_view = np.load(dog_viewing_filename)
        cat_view = cat_view.squeeze()
        dog_view = dog_view.squeeze()

        return cat_view, dog_view, X, Y

def interact(cat_viewing_data, dog_viewing_data):
    '''
    this function lets you view the dataset images
    :param cat_viewing_data:
    :param dog_viewing_data:
    :return:
    '''
    #create a loop for the user to view the images
    active=True
    while(active):
        active = view_cat_and_dog_training_data(cat_viewing_data, dog_viewing_data)