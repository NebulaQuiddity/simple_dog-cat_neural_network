<p align="center">
  <img src="https://user-images.githubusercontent.com/29058151/33108998-5f20e9a0-cefc-11e7-9a40-2c9148f7ff6c.png" width=700/>
</p>

[![YouTube](https://img.shields.io/badge/YouTube-tfs%20home-red.svg)](http:youtube.com/thingsfromspace)
[![Twitter](https://img.shields.io/badge/Twitter-%40tfs__space-blue.svg)](http://twitter.com/tfs_space)
[![Discord chat](https://img.shields.io/badge/Discord-discuss%20tfs%20%E2%86%92-orange.svg)](https://discord.gg/k4hu5G9)
[![Vimeo](https://img.shields.io/badge/Vimeo-more%20vids-brightgreen.svg)](https://vimeo.com/thingsfromspace)
[![Instagram](https://img.shields.io/badge/Instagram-%40things__from__space-lightgrey.svg)](https://www.instagram.com/things_from_space/)
[![Vidme](https://img.shields.io/badge/Vidme-the%20video%20alternative-yellow.svg)](https://vid.me/things_from_space)


# Learning from the classic MNIST dataset
This simple neural network attempts to classify MNIST images, as I am aware many, many, many people have already done. However, this project is primarily focused on solidifying my knowledge on feedforward neural networks before I move on to more complex learning models. 

# Benefits
  - Relatively easy to use
  - Well documented? I guess you can decide if that's true!
  - Honestly, that is this project's central benefit. I do not expect it to work faster or more accurately than the neural networks built by experts in the machine learning field, obviously. However, if you are also looking to develop a ffnn from scratch, looking at this project may be instructional. Furthermore, I would highly, HIGHLY, recommend checking out Michael Nielsen's online (and free!) [book](http://neuralnetworksanddeeplearning.com).
  
# So, how do I use it? Well, I'm glad you asked...
## Creating/loading data
The MNIST data is not particularly difficult to load, so if you have your own method of doing this, that's completely fine. Just make sure that the labels are formatted as an array of 10 by 1 vectors, not as a single scalar.

Otherwise, I suggest that you download the MNIST data as csv files from [this](https://pjreddie.com/projects/mnist-in-csv/) source. Following your download, you can run this to load the data into .npy files, which can later be loaded in quickly.
```
import load_mnist_data as lmnist
lmnist.create_mnist_data('tr_data.csv', 'te_data.csv', True, 
                         'trx.npy', 'try.npy', 'tex.npy', 'tey.npy')
``` 
This will create 4 .npy files that contain both the training and testing input and output data. In order to load this data, run 
```
trx, tr_y, tex, tey = lmnist.load_data_from_files(('trx.npy', 'try.npy','tex.npy', 'tey.npy'))
```
### Viewing images
If you are a bit impatient and want to check out some of the images, you can use 
```
lmnist.show_mnist_images([image array])
```
to view images in either the training or test set.
