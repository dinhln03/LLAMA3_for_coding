import tensorflow as tf
from current_net_conf import *


class RnnDropoutPlaceholders:
    probability = tf.placeholder(tf.float32, [])

    @staticmethod
    def feed(prob=1.0):
        return {RnnDropoutPlaceholders.probability: prob}


class MultiRnnWithDropout(tf.nn.rnn_cell.DropoutWrapper):

    def __init__(self, num_layers, state_size):
        internal_cells = [tf.nn.rnn_cell.GRUCell(state_size) for _ in range(num_layers)]
        internal_cell = tf.nn.rnn_cell.MultiRNNCell(internal_cells)
        super(MultiRnnWithDropout, self).__init__(
            cell=internal_cell,
            output_keep_prob=RnnDropoutPlaceholders.probability,
            state_keep_prob=RnnDropoutPlaceholders.probability
        )

    def initial_state(self, first_layer_initial_state):
        initial_state = self.zero_state(BATCH_SIZE, tf.float32)
        initial_state = list(initial_state)
        initial_state[0] = first_layer_initial_state
        initial_state = tuple(initial_state)
        return initial_state

    @staticmethod
    def zero_initial_inputs(size):
        return tf.zeros([BATCH_SIZE, size], dtype=tf.float32)


class EncoderDropoutPlaceholders:
    probability = tf.placeholder(tf.float32, [])

    @staticmethod
    def feed(prob=1.0):
        return {EncoderDropoutPlaceholders.probability: prob}


def apply_dropout_to_encoder_rnn_cells(cells):
    return [
        tf.nn.rnn_cell.DropoutWrapper(
            cell=cell,
            output_keep_prob=EncoderDropoutPlaceholders.probability,
            state_keep_prob=EncoderDropoutPlaceholders.probability
        )
        for cell in cells
    ]
