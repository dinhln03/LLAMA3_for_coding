# pass test
import numpy as np

def prepare_input(input_size):
    return [np.random.rand(input_size), np.random.rand(input_size)]

def test_function(input_data):
    return np.convolve(input_data[0], input_data[1])
