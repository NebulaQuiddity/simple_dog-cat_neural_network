# simple_image_classifier-using_ffnn
A feed forward neural network that aims to classify groups of images (specifically MNIST, in this case).
# Version 1: classifying dogs and cats
A simple, feed-forward neural network that attempts to classify images as either a cat or a dog. 
# Data
The data used in this example was from the online data science website Kaggle. You can view this competition and download the data here: https://www.kaggle.com/c/dogs-vs-cats/data
# Creating the Data
To load in the (training) data, you will need to have it in a folder named 'train'. In this folder, you should put the 25,000 images from the dataset. Do not change their names from ```cat.#.jpg``` and ```dog.#.jpg```. To create array files of the data on your computer, import ```load_image_data.py``` and run ```setup_kaggle_dog_and_cat_dataset(arguments)```. This will take a while, but once finished it will have created 2 or 4 files, depending on whether you chose to load the viewing dataset or not.
# Loading the data
Once you have created the data on you computer, you can run ```load_dog_and_cat_training_data``` to load the data into the program. This should take anywhere from 7 to 20 seconds. If you also loaded the viewing data, you can view the images by running ```interact(arguments)```.
