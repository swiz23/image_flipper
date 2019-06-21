# Image Flipper
Created by Solomon Wiznitzer as a fun independent project to understand how neural networks work. The code follows the approach presented by Dr. Andrew Ng in his online Coursera course - [Neural Networks and Deep Learning](https://www.coursera.org/learn/neural-networks-deep-learning). Note that the code is entirely my own but is loosely based on the implementation in some of the course projects.

## Description
**Objective:** To create a neural network from scratch (just using Python and Numpy) to parse a directory of images and reorient them if they are not right-side up.

**Overview:** This package is meant to be a tutorial on implementing a binary classifier neural network. It is aimed towards those who understand the basic theory. There were three parts to this project.
- *Coding a neural network* - This part focused on coding up the functions needed to implement a neural network - specifically forward propagation, computation of the loss function, backward propagation, and updating the weights and biases.
- *Tuning the neural network* - After creating the functions, the next agenda was to adjust the hyper-parameters (like learning rate, number of hidden layers, number of hidden units in each layer, and the type of activation functions for each layer) to obtain the best results.
- *Pre and post-processing of images* - This step involved preprocessing the input images to a standard size that could then be fed into the neural network. It also involved rotating the images in the output directory to the proper orientation based on the results from the neural network.

**System Info:** This package was tested on Ubuntu Linux 16.04 using Python 2.7.12.

### Runtime Instructions
After cloning the repo, just type `python nn_main.py` into the terminal to get started. You will be prompted with a series of questions on how you would like to design your network. The first question you will see is:
```
Enter the number of layers you would like in your neural network (positive integer):
```
For whatever reason, when specifying the number of layers in a neural network, the convention is to exclude the input layer. As a result, the arbitrary network in the image below can be described as a three-layer network as it has two hidden layers and one output layer.
![nn_representation](media/nn.png)
The same standard should be used when specifying the number of layers in this program.
Next, you will be prompted to:
```
Enter the number of units you would like in layer #1 (positive integer):
```
In the example above, layer #1 corresponds to the column of blue circles. Each circle represents a unit so there are four units in this layer.
The program will then prompt you with:
```
Enter the activation function that will be applied to all units in layer #1 ("sigmoid", "tanh", "relu", "leaky_relu"):
```
Here, you have the option to choose between four different activation functions that will be applied to each unit in the layer: `ReLu`, `Leaky ReLu`, `Sigmoid`, and `Tanh`. Note that this program does not allow you to customize the activation function for each unit in a given layer - it only allows you to customize the activation function that will be used for all units in the layer.
The above two questions will repeat for the rest of the hidden layers and output layer. Note that since the model is meant to be a binary classifier, the user should always input `1` and `"sigmoid"` for the output layer.
Next, you will be asked:
```
Enter the learning rate (positive float):
```
The learning rate represents the length of each step taken by gradient descent for each iteration. Values should range between 0 and 1 but are typically less than 0.1. Too large of a value will decrease the probability that a minimum will be found in the cost function. Too small a value and it will take a much longer time for gradient descent to find that minimum.
Finally, you will asked to asked:
```
Enter the number of iterations (positive integer):
```
This is where you should input the number of gradient descent iterations the neural network should do while training. A large value will take more time while a small value might not be enough to find the minimum of the cost function. This can be easily gauged by looking at the output of the cost function on the graph that pops up or in the terminal as the network is training.
