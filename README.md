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

Note that you can also comment out these prompts in the `nn_main.py` file and instead hard-code your model directly. The sections to comment and uncomment are clearly labeled in the file.

### Structure
 The directory structure showing how the images were organized is shown below. For my application, images that were oriented correctly were placed in the `correct_images` directory while those that were improperly oriented were placed in the `incorrect_images` directory.
 ```
 .
 ├── test
 │   └── images
 │       ├── Test Image 1
 │       ├── Test Image 2
 │       └── Test Image ...
 ├── train
 │   ├── correct_images
 │   │   ├── Train Correct Image 1
 │   │   ├── Train Correct Image 2
 │   │   └── Train Correct Image ...
 │   └── incorrect_images
 │       ├── Train Incorrect Image 1
 │       ├── Train Incorrect Image 2
 │       └── Train Incorrect Image ...
 └── validate
     ├── correct_images
     │   ├── Validate Correct Image 1
     │   ├── Validate Correct Image 2
     │   └── Validate Correct Image ...
     └── incorrect_images
         ├── Validate Incorrect Image 1
         ├── Validate Incorrect Image 2
         └── Validate Incorrect Image ...
```

### Other Applications
While I was interested in rotating images you would normally see while hiking, it wouldn't be too difficult to do the same for a different application. For example, you could train the network to recognize the correct orientation of images with skyscrapers or people! Just swap out the pictures currently in the image directories for your own, and you'll be good to go! With a bit of code tweaking on the 'post image-processing' side, you could also train a network to recognize when there are people in images. Then, you could parse a directory of images and separate them into two directories - one with people and one without.

### Future Work
At some point, I plan to add the ability to make a network be able to do multi-classification. Additionally, I might add a feature to just print out to the console whether an image belongs to one class or the other.
