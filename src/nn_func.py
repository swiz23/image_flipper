import numpy as np
import matplotlib.pyplot as plt
import os, sys
from PIL import Image
import glob
# 'm' is the number of examples in the dataset
# 'n_x' is the number of input features in each example
# 'n_h' is the number of hidden layers in the neural net (includes output layer)

class neural_net:
    def __init__(self, layers, layer_activations):
        # dimensions of 'X' are (n_x, m)
        # dimensions of 'Y' are (1, m)
        # dimensions of 'layers' is (1, n_h)
        # 'lr' is the learning rate
        # 'iterations' is the number of gradient descent iterations
        # 'layer_activations' is a list of what activations to use for each layer where the
        # first input corresponds to the output of the first hidden layer, etc...
        self.layers = layers
        self.layer_activations = layer_activations
        self.layer_activations.insert(0, "none")
        self.L = None
        np.random.seed(2)

    def predict(self, params, X, Y = None):
        A_dict = forward_model(X, params, self.layer_activations, self.L)
        AL = A_dict["A" + str(self.L - 1)]
        predictions = (AL > 0.7)
        percent_accuracy = None
        if Y is not None:
            # this should be all zeros if there is no difference between the predictions and Y
            p = np.logical_xor(predictions, Y)
            percent_accuracy = str(np.round(100 - 100.0 * np.sum(p)/Y.shape[1], 1)) + '%'
        return predictions, percent_accuracy

    def flip(self, params, img_dict):
        path = '../test/images/*'
        for infile in glob.glob(path):
            ar_im = img_dict[infile]
            image_list = [ar_im, np.rot90(ar_im)]
            image_list.append(np.rot90(image_list[1]))
            image_list.append(np.rot90(image_list[2]))
            image_array = np.array(image_list)
            image_flatten = image_array.reshape(4, -1).T
            X = image_flatten/255.0
            A_dict = forward_model(X, params, self.layer_activations, self.L)
            AL = np.squeeze(A_dict["A" + str(self.L - 1)])
            max_ind = np.argmax(AL)
            if AL[max_ind] > 0.7:
                try:
                    image = Image.open(infile)
                    image = image.rotate(max_ind*90, expand=1)
                    image.save(infile, "JPEG")
                    print infile.split('images/')[1] + " was rotated counterclockwise by " + str(max_ind*90) + " degrees."
                except IOError:
                    print "cannot create JPG for '%s'" % infile
            else:
                print infile.split('images/')[1] + " was not rotated since the confidence was too low."

    def train(self, X, Y, lr, iterations):
        costs = []
        self.layers.insert(0, X.shape[0])
        self.L = len(self.layers)
        params = init_params(self.layers, self.L)
        print "\n-------------------- Training --------------------"
        for i in range(iterations):
            A_dict = forward_model(X, params, self.layer_activations, self.L)
            AL = A_dict["A" + str(self.L - 1)]
            loss, dL_dAL = compute_loss(AL, Y)
            grad_dict, params = backward_model(dL_dAL, A_dict, params, self.layer_activations, lr, self.L)
            if iterations < 100 or i % int(0.01*iterations) == 0 or i == iterations-1:
                costs.append(loss)
            if i % int(0.1*iterations) == 0 or i == iterations-1:
                print("Cost after iteration %5i: %f" %(i, loss))
        plt.plot(np.linspace(0, iterations, len(costs)), np.squeeze(costs))
        plt.ylabel('Loss')
        plt.xlabel('Iterations of Gradient Descent')
        plt.title("Learning rate=" + str(lr))
        return params

# Initialize weight matrices to small random values and bias vectors to zero
def init_params(layers, num_layers):
    # Weight matrices should have dimensions of (num neurons in layer, num neurons in last layer)
    # Bias vectors should have dimensions of (num neurons in layer, 1)
    params = {}
    for l in range(1, num_layers):
        params["W" + str(l)] = np.random.randn(layers[l], layers[l-1]) * 0.01
        params["b" + str(l)] = np.zeros((layers[l], 1))
        layer_prev = layers[l]
    return params

def forward_model(X, params, layer_activations, num_layers):
    A_dict = {"A0": X}
    for l in range(1, num_layers):
        A = forward_prop(A_dict["A" + str(l-1)], params["W" + str(l)], params["b" + str(l)], layer_activations[l])
        A_dict["A" + str(l)] = A
    return A_dict

def compute_loss(AL, Y):
    # AL is the activation output from the last neuron that is used to compute the loss.
    # AL has a shape of (1, m)
    # Y is the correct classification of the training examples. It has a shape of (1, m)
    # The cost function is calculated using the Cross Entropy Loss method
    m = AL.shape[1]
    Loss = -1.0/m * (np.dot(Y, np.log(AL.T)) + np.dot(1 - Y, np.log(1 - AL.T)))
    Loss = np.asscalar(Loss)
    # Now, compute dL_dAL which will be passed to the backward_prop function
    dL_dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    return Loss, dL_dAL

def backward_model(dL_dAL, A_dict, params, layer_activations, lr, num_layers):
    grad_dict = {"dL_dA" + str(num_layers - 1):dL_dAL}
    for l in range(num_layers - 1, 0, -1):
        grad_dict["dL_dA" + str(l-1)], grad_dict["dL_dW" + str(l)], grad_dict["dL_db" + str(l)] = backward_prop(grad_dict["dL_dA" + str(l)], A_dict["A" + str(l)], A_dict["A" + str(l - 1)], params["W" + str(l)], params["b" + str(l)], layer_activations[l])
        # update weights
        params["W" + str(l)] -= lr * grad_dict["dL_dW" + str(l)]
        params["b" + str(l)] -= lr * grad_dict["dL_db" + str(l)]

    return grad_dict, params

def load_pics(image_type):
    thumb_size = 128, 128
    image_list = []
    image_dict = {}
    Y = []
    image_directories = ["incorrect_images", "correct_images"]
    if image_type == "train":
        folder = "train"
    elif image_type == "validate":
        folder = "validate"
    elif image_type == "test":
        folder = "test"
        image_directories = ["images"]
    for d in range(len(image_directories)):
        path = '../'+folder+'/' + image_directories[d] + '/*'
        for infile in glob.glob(path):
            try:
                im = Image.open(infile)
                size = im.size
                if size[0] > size[1]: # if the width of the pic is larger than the height
                    region = im.crop((int(size[0]/2. - size[1]/2.), 0, int(size[0]/2. + size[1]/2.), size[1]))
                elif size[0] < size[1]: # if the width of the pic is smaller than the height
                    region = im.crop((0, int(size[1]/2. - size[0]/2.), size[0], int(size[1]/2. + size[0]/2.)))
                else:
                    region = im
                region.thumbnail(thumb_size)
                assert region.size == thumb_size
                ar_im = np.array(region)
                if ar_im.shape != (128, 128, 3):
                    print "Wrong shape"
                image_list.append(ar_im)
                image_dict[infile] = ar_im
                Y.append(1 * d)
            except IOError:
                print "cannot open image for '%s'" % infile
    image_array = np.array(image_list)
    image_flatten = image_array.reshape(image_array.shape[0], -1).T
    X = image_flatten/255.0
    Y = np.array(Y).reshape(1, len(Y))
    assert Y.shape == (1, X.shape[1])
    return X, Y, image_dict

def sigmoid_back(dL_dA, A):
    # Calculate dL_dZ = dL_dA * dA_dZ   ->  where dA_dZ is the derivative of A wrt Z
    # The derivative of a sigmoid function is A * (1 - A)
    dA_dZ = A * (1 - A)
    dL_dZ = dL_dA * dA_dZ
    return dL_dZ

def tanh_back(dL_dA, A):
    # Calculate dL_dZ = dL_dA * dA_dZ   -> where dA_dZ is the derivative of A wrt Z
    # The derivative of a tanh function is 1 - A^2
    dA_dZ = 1 - np.power(A, 2)
    dL_dZ = dL_dA * dA_dZ
    return dL_dZ

def relu_back(dL_dA, A):
    # Calculate dL_dZ = dL_dA * dA_dZ   -> where dA_dZ is the derivative of A wrt Z
    # The derivative of a relu function is 1 if A > 0 else 0. Thus, we can just
    # replace negative numbers in dL_dA with zeros and keep everything else the same
    # (equivalent to multiplying by 1)
    dL_dZ = np.array(dL_dA, copy=True)
    dL_dZ[A == 0] = 0
    return dL_dZ

def leaky_relu_back(dL_dA, A):
    # Calculate dL_dZ = dL_dA * dA_dZ   -> where dA_dZ is the derivative of A wrt Z
    # The derivative of a leaky relu function is 1 if A > 0 else 0.01. Thus, we can just
    # multiply negative numbers in dL_dA with 0.01 and keep everything else the same
    # (equivalent to multiplying by 1)
    dA_dZ = np.ones(dL_dA.shape)
    dA_dZ[A == 0.01] = 0.01
    dL_dZ = dL_dA * dA_dZ
    return dL_dZ

def sigmoid(Z):
    A = 1/(1+np.exp(-Z))
    return A

def tanh(Z):
    A = np.tanh(Z)
    return A

def relu(Z):
    A = np.maximum(0.0, Z)
    return A

def leaky_relu(Z):
    A = np.array(Z, copy=True)
    A[Z <= 0] = 0.01
    return A

def forward_prop(A_prev, W, b, activation):
    # A_prev has shape of (num units in last layer, m)
    # W has shape of (num units in layer, num units in last layer)
    # b has shape of (num units in layer, 1)
    # activation is either 'sigmoid', 'relu', or 'tanh'

    Z = np.dot(W, A_prev) + b
    if activation == "sigmoid":
        A = sigmoid(Z)
    elif activation == "tanh":
        A = tanh(Z)
    elif activation == "relu":
        A = relu(Z)
    elif activation == "leaky_relu":
        A = leaky_relu(Z)

    return A

def backward_prop(dL_dA, A, A_prev, W, b, activation):
    # dL_dA should have same shape as A
    # A is the activation matrix of the current layer with shape (num units in layer, m)
    # activation is either 'sigmoid', 'relu', or 'tanh'
    if activation == "sigmoid":
        dL_dZ = sigmoid_back(dL_dA, A)
    elif activation == "tanh":
        dL_dZ = tanh_back(dL_dA, A)
    elif activation == "relu":
        dL_dZ = relu_back(dL_dA, A)
    elif activation == "leaky_relu":
        dL_dZ = leaky_relu_back(dL_dA, A)

    # Calculate the gradient of dL_dA_prev = dZ_dA_prev * dL_dZ.
    # We know Z = W * A_prev + b. So dZ_dA_prev = W.
    # dL_dA_prev should have a shape of (num hidden units in last layer, m).
    # The shape of W.T gives (num hidden units in last layer, num hidden units in current layer).
    # The shape of dL_dZ is (num hidden units in current layer, m) so this calculation works out.
    dL_dA_prev = np.dot(W.T, dL_dZ)
    m = A_prev.shape[1]
    # Now calculate the gradient of dL_dW = dL_dZ * dZ_dW.
    # We know Z = W * A_prev + b. So dZ_dW = A_prev
    # dL_dZ has a shape of (num hidden units in current layer, m)
    # A_prev has a shape of (num hidden units in last layer, m)
    # so we must do A_prev.T to make the dot product work.
    # This results in dL_dW having a shape of
    # (num hidden units in current layer, num hidden units in last layer)
    # which matches W as expected. Then we take the average by dividing by 'm'
    dL_dW = 1.0/m * np.dot(dL_dZ, A_prev.T)
    # Now calculate the gradient of dL_db = dL_dZ * dZ_db.
    # We know Z = W * A_prev + b. So dZ_db = 1, so dL_db is...
    # the sum of all biases across each example divided by m
    # the shape of dL_db is (num hidden units in current layer, 1)
    dL_db = 1.0/m * np.sum(dL_dZ, axis = 1, keepdims = True)

    return dL_dA_prev, dL_dW, dL_db
