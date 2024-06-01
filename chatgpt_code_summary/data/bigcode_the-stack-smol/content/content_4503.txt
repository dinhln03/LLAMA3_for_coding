
import numpy as np

class Perceptron(object):

    def __init__(self, input_num, activator):

        self.activator = activator
        self.weights = np.zeros((input_num))
        self.bias = 0.0

    def __str__(self):

        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        
        return self.activator(np.dot(input_vec, self.weights) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):

        for _ in range(iteration):
            self._one_iteration(input_vecs, labels, rate)

    def _one_iteration(self, input_vecs, labels, rate):

        samples = zip(input_vecs, labels)

        for input_vec, label in samples:

            output = self.predict(input_vec)

            self._update_weight(input_vec, output, label, rate)

    def _update_weight(self, input_vec, output, label, rate):

        delat = label - output
        self.weights += rate * delat * input_vec
        self.bias += rate * delat


def f(x):

    if x > 0: return 1
    else:  return 0


def get_train_dataset():

    vecs = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    labels = np.array([1, 0, 0, 0])

    return vecs, labels


def train_and_perceptron():

    p = Perceptron(2, f)

    input_vecs, labels = get_train_dataset()
    p.train(input_vecs, labels, 10, 0.1)

    return p


if __name__ == "__main__":

    and_perceptron = train_and_perceptron()
    print(and_perceptron)

    print ('1 and 1 = ' , and_perceptron.predict([1, 1]))
    print ('1 and 0 = ' , and_perceptron.predict([1, 0]))
    print ('0 and 1 = ' , and_perceptron.predict([0, 1]))
    print ('0 and 0 = ' , and_perceptron.predict([0, 0]))

