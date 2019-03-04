<p align="center">
  <img src="https://user-images.githubusercontent.com/29058151/33108998-5f20e9a0-cefc-11e7-9a40-2c9148f7ff6c.png" width=700/>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/29058151/33245815-03bd877c-d2ca-11e7-9330-56fd6ee04449.png" width=700/>
</p>

# Learning from the classic MNIST dataset
This simple neural network attempts to classify MNIST images, as I am aware many, many, many people have already done. However, this project is primarily focused on solidifying my knowledge on feedforward neural networks before I move on to more complex learning models. 

# Benefits
  - Relatively easy to use
  - Well documented? I guess you can decide if that's true!
  - Honestly, that is this project's central benefit. I do not expect it to work faster or more accurately than the neural networks built by experts in the machine learning field, obviously. However, if you are also looking to develop a ffnn from scratch, looking at this project may be instructional. Furthermore, I would highly, HIGHLY, recommend checking out Michael Nielsen's online (and free!) [book](http://neuralnetworksanddeeplearning.com).
  
# So, how do I use it? Well, I'm glad you asked...
## Creating/loading data
The MNIST data is not particularly difficult to load, so if you have your own method of doing this, that's completely fine. Just make sure that the labels are formatted as an array of 10 by 1 vectors, not as a single scalar.

Otherwise, I suggest that you download the MNIST data as csv files from [this](https://pjreddie.com/projects/mnist-in-csv/) source. Following your download, you can run this to load the data into .npy files, which can later be imported into the project quickly.
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

# Creating a neural network
This project makes creating a neural network simple and easy. All that you need to run is 
```
import feed_forward_neural_network as ffnn
network = ffnn.FeedForwardNeuralNetwork(trx, tr_y, tex, tey, [network shape])
```
Network shape is the shape of your network without the input layer, since this can already be determined from the input data. For example, if you wanted to create a network with the shape [784 (input), 300, 100, 50, 10], network shape would be a tuple/list `(300, 100, 50, 10)`.

# Classification
## Training

You can train the neural network by running:
```
network.train(batch_size, iterations, learning_rate, hidden_layer_activation_function, output_layer_activation_function, weights_file_name, biases_file_name, (True, 10), epsilon)
```
where:
- `batch_size` is the batch size (duh)
- `iterations` is the number of training iterations you want to run
- `learning_rate` is the learning rate (duh again)
- `hidden_layer_activation_function` and `output_layer_activation_function` are the activation functions that you want to use for the hidden and output layers. For example, `'lr', 's'` would mean that the hidden layers of the neural network will use the leaky relu activation function and the output layer will use the sigmoid activation function.
- `weights_file_name` and `biases_file_name` are the names of the files where the weights and biases of this network will be saved (and thus they should end in '.npy')
- `(True, 10)` indicates that the cost of the network should be printed every 10 iterations
- `epsilon` is the epsilon for the leaky relu activation function (leave as `None` if you don't use leaky relu)


## Testing
### Visual Testing Over\ the Training Set
One way that you can visually look for errors in your neural network is by observing the images that it classifies correctly and incorrectly. This projects allows you to observe visually observe the images that your network messes up on in the training set (! note that some models may overfit the data so that they classify virtually every image in the training set correctly, though they may still mess up in the test set).

You can test the network visually by running:
 ```
 network.test_network_visually(network.input_X, network.output_Y)
 ```
<p>
  <img src="https://user-images.githubusercontent.com/29058151/35200222-5ef1968c-fec7-11e7-9b46-280cbd5ff38e.png"
       </p>

### Testing on the Test Set
In order to properly measure the accuracy of your model, you can use the test set (which the model has never 'seen'). To do this, run:
```
correctly_classified = network.test_network()
```
## Loading a Previously Trained Network
### Loading Weights and Biases
In order to load weights and biases from a previous model, run:
```
network.load_weights_and_biases('weights.npy', 'biases.npy')
```
### Setting Other Parameters
The network doesn't save your parameters, so you will need to change them manually. For example:
```
network.hidden_layer_activation_function = 'lr'
network.output_layer_activation_function = 's'
network.epsilon = 0.01
```
