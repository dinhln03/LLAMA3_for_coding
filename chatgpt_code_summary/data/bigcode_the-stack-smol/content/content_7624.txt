'''
DCGAN - MNIST Grayscale Handwritten Digits.

Ref: https://machinelearningmastery.com/generative_adversarial_networks/
'''

# import tensorflow.python.ops.numpy_ops.np_config
from data import loadRealSamples
from generator import createGenerator
from discriminator import createDiscriminator
from gan import createGan, train

if __name__ == '__main__':
    latentDim = 100
    dataset = loadRealSamples()
    discriminator = createDiscriminator()
    generator = createGenerator(latentDim)
    gan = createGan(discriminator, generator)
    train(discriminator, generator, gan, dataset, latentDim)
