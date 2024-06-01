# some utils taken from the DeepXplore Implementation

import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.preprocessing import image
from keras import models, layers, activations

from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from itertools import combinations


#loads a mnist image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28), grayscale=True)
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data.reshape(1, 28, 28, 1)

    input_img_data = input_img_data.astype('float32')
    input_img_data /= 255
    # input_img_data = preprocess_input(input_img_data)  # final input shape = (1,224,224,3)
    return input_img_data

def init_neuron_cov_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    return layer_name, index


def get_neuron_coverage(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_neuron_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True
                print("new coverage found")


#To test

#gets the distance of the points in standard deviations
#note that it assumes that the points are normally distributed
def distance(point, mean, covarianceMatrix):
    return mahalanobis(point, mean, inv(covarianceMatrix))

# an adaptation of some code from deepXplore
# initializes a dictionary that will store which qudrants have been covered
# model - the model we are looking to covered
# layer_index - the layer we are exploring
# group_size - size of the group of neurons we are analyzing
# model_layer_dict - the object we want to initialize
def init_orthant_cov_dict(model, layer_index, group_size, model_layer_dict):
    layer = model.layers[layer_index]
    # some error handling
    if 'flatten' in layer.name or 'input' in layer.name:
        print("error in init_dict: layer_index points to the wrong layer")
    # we initialize each combination
    for neuron_group in combinations(range(layer.output_shape[-1]), group_size): # layer.output_shape[-1] returns the number of total_neurons
        for orthant in range(2^group_size-1):
            model_layer_dict[(neuron_group, orthant)] = False

def get_orthant_coverage(model_layer_dict):
    covered_orthants = len([v for v in model_layer_dict.values() if v])
    total_orthants = len(model_layer_dict)
    return covered_orthants, total_orthants, covered_orthants / float(total_orthants)

#this is meant to pick a orthant that is not covered
# we actually don't need to use this just yet, maybe if I decide to implement for DeepXplore
def next_orthant_to_cover(model_layer_dict):
    not_covered = [(neuron_group, orthant) for (neuron_group, orthant), v in model_layer_dict.items() if not v]
    if not_covered:
        neuron_group, orthant = random.choice(not_covered)
    else:
        neuron_group, orthant = random.choice(model_layer_dict.keys())
    return neuron_group, orthant


# creates a shortened model that ends at the nth layer, and has no activation function
# same code as from collect_data
def create_shortened_model(model, layer_depth):
    # we get the neuron output for the penultimate layer for each neuron

    # implemented with help from the suggestion at: https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
    # we recreate the model, delete layers up to and including the layer we want to analyze, add a blank layer with no activation, and then import the old weights to this layer.

    #make a new model

    # some simple input checks
    if(layer_depth < 0):
        println ('layer depth must be positive!')
        sys.exit()

    if(layer_depth > len(model.layers)):
        println ('layer depth too large!')
        sys.exit()

    # save the original weights
    wgts = model.layers[layer_depth].get_weights()
    nthLayerNeurons = model.layers[layer_depth].output_shape[1]

    #remove layers up to the nth layer
    for i in range(len(model.layers)-layer_depth):
        model.pop()
    model.summary
    # add new layer with no activation
    model.add(layers.Dense(nthLayerNeurons,activation = None))

    # with the new layer, load the previous weights
    model.layers[layer_depth].set_weights(wgts)

    # get the output of this new model.
    return Model(inputs=model.input, outputs=model.layers[layer_depth].output )

#this code updates the coverage given a certain input
def update_orthant_coverage(input_data, shortened_model, model_layer_dict, mean_vector, covariance_matrix, group_size=1, sd_threshold=1):

    layer_outputs = shortened_model.predict(input_data) #get the output
    # the reason that we use layer_outputs[0] is change it into a single row, rather than an array with a row.

    for neuron_group in combinations(range(layer_outputs.shape[-1]),group_size):
        group_output = np.asarray([layer_outputs[0][i] for i in neuron_group]) #get a list of the outputs

        # we do binary addition to get the correct orthant index.
        # for example, if we only have a 2 variables, we have 4 quadrants. we need to classify into 0,1,2,3 index
        #init the tools to find which orthant is being explored
        orthant = 0
        add = int(1)
        for neuron_index in neuron_group:
            if layer_outputs[0][neuron_index] > mean_vector[neuron_index]:
                orthant += add
            add *= 2

        if model_layer_dict[(neuron_group,orthant)] == True:
            continue #don't do the expensive action of loading the group cov, group mean, and calculating the distance

        group_mean = np.asarray([mean_vector[i] for i in neuron_group]) #list of mean
        #initialize the group numpy array for later calculation
        group_cov_matrix = np.asarray([[covariance_matrix[j][i] for i in neuron_group] for j in neuron_group]) #dont ask me why

        if(distance(group_output, group_mean, group_cov_matrix)>sd_threshold):
            model_layer_dict[(neuron_group,orthant)] = True

# just a simple check if we have full coverage works for any coverage
def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


# from here on is code from deepxplore

# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
    start_point[1]:start_point[1] + rect_shape[1]] = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                                                     start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    return grad_mean * new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (
        random.randint(0, gradients.shape[1] - rect_shape[0]), random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0], start_point[1]:start_point[1] + rect_shape[1]]
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
        start_point[1]:start_point[1] + rect_shape[1]] = -np.ones_like(patch)
    return new_grads


def init_coverage_tables(model1, model1_layer_index, model2, model2_layer_index, model3, model3_layer_index, group_size = 1):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model1_layer_index, group_size, model_layer_dict1)
    init_dict(model2, model2_layer_index, group_size, model_layer_dict2)
    init_dict(model3, model3_layer_index, group_size, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3

def init_neuron_coverage_table(model1):
    model_layer_dict1 = defaultdict(bool)
    init_neuron_cov_dict(model1, model_layer_dict1)
    return model_layer_dict1

def init_orthant_coverage_table(model1, layer_index, group_size):
    model_layer_dict1 = defaultdict(bool)
    init_orthant_cov_dict(model1, layer_index, group_size, model_layer_dict1)
    return model_layer_dict1

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
