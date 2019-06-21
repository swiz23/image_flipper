import matplotlib.pyplot as plt
import nn_func as nn
import numpy as np

def main():
    # Define hyperparameters:
    #   number of layers
    #   number of units in each layer
    #   type of activation funtion to be applied in each layer
    #   learning rate for gradient descent
    #   number of gradient descent iterations.

    # number of desired layers in the neural network (not including the input layer)
    num_layers = int(input("Enter the number of layers you would like in your neural network (positive integer): "))

    # layers - number of hidden units in each layer excluding the input layer
    # ex. [4, 1] means there are 4 hidden units in the first layer and 1 unit in the output layer
    layers = []

    # layer_activations - activation function in each layer excluding the input layer
    # ex. ["relu", "sigmoid"] means that the 'ReLu' activation function is used at the first layer's output
    # and the 'Sigmoid' activation function is used at the output of the second layer
    layer_activations = []

    for x in range(1, num_layers+1):
        num_units = int(input("Enter the number of units you would like in layer #{} (positive integer): ".format(x)))
        layer_type = input("Enter the activation function that will be applied to all units in layer #{} (\"sigmoid\", \"tanh\", \"relu\", \"leaky_relu\"): ".format(x))
        layers.append(num_units)
        layer_activations.append(layer_type)

    # lr - learning rate
    # ex. lr = 0.015
    lr = float(input("Enter the learning rate (positive float): "))

    # iterations - number of iterations of gradient descent
    # ex. iterations = 300
    iterations = int(input("Enter the number of iterations (positive integer): "))

    # Model description
    print "\n-------------------- Model Description --------------------"
    print "Number of hidden units in each layer: " + str(layers)
    print "Type of activation function in each layer: " + str(layer_activations)
    print "Learning rate: " + str(lr)
    print "Number of iterations: " + str(iterations)

    # Initialize model
    model_1 = nn.neural_net(layers, layer_activations)

    # Train
    X_train, Y_train, _ = nn.load_pics("train")
    params = model_1.train(X_train, Y_train, lr, iterations)
    pred_train, p_train = model_1.predict(params, X_train, Y_train)
    print ""
    print "The actual classification of the training images are:    " + np.array2string(np.squeeze(Y_train))
    print "The predicted classification of the training images are: " + np.array2string(np.squeeze(pred_train.astype(int)))
    print "The overall accuracy of the model is: " + p_train

    # Validate
    X_val, Y_val, _ = nn.load_pics("validate")
    pred_val, p_val = model_1.predict(params, X_val, Y_val)
    print "\n-------------------- Validating --------------------"
    print "The actual classification of the validating images are:    " + np.array2string(np.squeeze(Y_val))
    print "The predicted classification of the validating images are: " + np.array2string(np.squeeze(pred_val.astype(int)))
    print "The overall accuracy of the model is: " + p_val

    # Test
    print "\n-------------------- Testing --------------------"
    X_test, _, image_dict_test = nn.load_pics("test")
    pred_test, _ = model_1.predict(params, X_test)
    print "The predicted classification of the testing images are:   " + np.array2string(np.squeeze(pred_test.astype(int)))
    model_1.flip(params, image_dict_test)

    plt.show()

if __name__ == '__main__':
    main()
