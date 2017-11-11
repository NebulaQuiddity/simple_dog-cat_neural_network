# Simple Image Classifier Using a Feed Forward Neural Network
A feed forward neural network that aims to classify groups of images.
# Version 1: Classifying MNIST Images
A simple, feed-forward neural network that attempts to classify images as a number between 0 and 9. 
# Data
In this example the classic MNIST image data was used, downloaded from this source (as a csv): https://pjreddie.com/projects/mnist-in-csv/
# Creating/Loading the Data
In order to create the data, you need to import `load_mnist_data` and run def `create_mnist_data(arguments)` (view the code for a full descriptions of the arguments). This function will either directly return the arrays or will save them to `.npy` files. To load in the arrays in the files, run `load_data_from_files(arguments)` (this will be quicker if to load than running `create_mnist_data()` every time).
