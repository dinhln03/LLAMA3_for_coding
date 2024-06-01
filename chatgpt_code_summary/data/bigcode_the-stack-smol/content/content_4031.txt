# reimplementation of https://github.com/guillaumegenthial/tf_ner/blob/master/models/lstm_crf/main.py

import functools
import json
import logging
from pathlib import Path
import sys
import numpy as np
import tensorflow as tf
# tf.enable_eager_execution()
from tf_metrics import precision, recall, f1

DATADIR = "../../../data/conll/"

# Setup Logging
Path('results').mkdir(exist_ok=True)
tf.logging.set_verbosity(logging.INFO)
handlers = [ logging.FileHandler('results/main.log'), logging.StreamHandler(sys.stdout)]
logging.getLogger('tensorflow').handlers = handlers

# Data Pipeline
def parse_fn(line_words, line_tags):
    """Encodes words into bytes for tensor

    :param line_words: one line with words (aka sentences) with space between each word/token
    :param line_tags: one line of tags (one tag per word in line_words)
    :return: (list of encoded words, len(words)), list of encoded tags
    """

    words = [w.encode() for w in line_words.strip().split()]
    tags = [t.encode() for t in line_tags.strip().split()]
    assert len(words) == len(tags), "Number of words {} and Number of tags must be the same {}".format(len(words), len(tags))
    return (words, len(words)), tags

def generator_fn(words_file, tags_file):
    """Enumerator to enumerate through words_file and associated tags_file one line at a time

    :param words_file: file path of the words file (one sentence per line)
    :param tags_file: file path of tags file (tags corresponding to words file)
    :return enumerator that enumerates over the format (words, len(words)), tags one line at a time from input files.
    """

    with Path(words_file).open('r') as f_words, Path(tags_file).open('r') as f_tags:
        for line_words, line_tags in zip(f_words, f_tags):
            yield parse_fn(line_words, line_tags)


def input_fn(words_file, tags_file, params = None, shuffle_and_repeat = False):
    """Creates tensorflow dataset using the generator_fn

    :param words_file: file path of the words file (one sentence per line)
    :param tags_file: file path of tags file (tags corresponding to words file)
    :param params: if not None then model hyperparameters expected - 'buffer' (as in buffer size) and 'epochs'
    :param shuffle_and_repeat: if the input is to be shuffled and repeat-delivered (say per epoch)
    :return: instance of tf.data.Dataset
    """

    params = params if params is not None else {}

    # shapes are analogous to (list of encoded words, len(words)), list of encoded tags
    shapes = (([None], ()), [None])
    types = ((tf.string, tf.int32), tf.string)

    defaults = (('<pad>', 0), 'O')

    generator = functools.partial(generator_fn, words_file, tags_file)
    dataset = tf.data.Dataset.from_generator(generator, output_shapes = shapes, output_types = types)

    if shuffle_and_repeat:
        dataset = dataset.shuffle(params['buffer']).repeat(params['epochs'])

    dataset = dataset.padded_batch(params.get('batch_size', 20), shapes, defaults).prefetch(1)\

    return dataset

def model_fn(features, labels, mode, params):
    """

    :param features: words from sentence and number of words per sentence
    :param labels: One tag per word
    :param mode:  tf.estimator.ModeKeys.TRAIN or  tf.estimator.ModeKeys.PREDICT or  tf.estimator.ModeKeys.EVAL
    :param params: dictionary of hyper parameters for the model
    :return:
    """

    # For serving, features are a bit different
    if isinstance(features, dict):
        features = features['words'], features['nwords']

    # Read vocab_words_file, vocab_tags_file, features
    words, nwords = features
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    vocab_words = tf.contrib.lookup.index_table_from_file(params['vocab_words_file'], num_oov_buckets = params['num_oov_buckets'])

    '''
    If the file contains the following: 
    B-LOC
    B-PER
    O
    I-LOC
    
    then indices = [0, 1, 3] and num_tags = 4
    
    Open Question: The special treatment of tag indices is probably needed for microavg metrics. Why though?
    '''

    with Path(params['vocab_tags_file']).open('r') as f:
        indices = [idx for idx, tag in enumerate(f) if tag.strip() != 'O']
        num_tags = len(indices) + 1

    # Word Embeddings
    # remember - as per the parse function "words" is a python list of
    word_ids = vocab_words.lookup(words)
    glove = np.load(params['glove'])['embeddings']
    glove = np.vstack([glove, [[0.]*params['dim']]])
    variable = tf.Variable(glove, dtype=tf.float32, trainable=False)
    embeddings = tf.nn.embedding_lookup(variable, word_ids)
    dropout = params['dropout']
    embeddings = tf.layers.dropout(embeddings, rate = dropout, training = training)

    # LSTM CRF
    time_major = tf.transpose(embeddings, perm = [1, 0, 2])
    lstm_cell_fw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.LSTMBlockFusedCell(params['lstm_size'])
    lstm_cell_bw = tf.contrib.rnn.TimeReversedFusedRNN(lstm_cell_bw)

    """
    Any LSTM Cell returns two things: Cell Output (h) and Cell State (c)

    Following this, lstm_fw or lstm_bw each return a pair containing:

    Cell Output: A 3-D tensor of shape [time_len, batch_size, output_size]
    Final state: a tuple (cell_state, output) produced by the last LSTM Cell in the sequence.

    """
    output_fw,_ = lstm_cell_fw(time_major, dtype = tf.float32, sequence_length = nwords)
    output_bw,_ = lstm_cell_bw(time_major, dtype = tf.float32, sequence_length = nwords)
    output = tf.concat([output_fw, output_bw], axis=-1)
    output = tf.transpose(output, perm=[1, 0, 2])
    output = tf.layers.dropout(output, rate=dropout, training=training)

    # CRf
    logits = tf.layers.dense(output, num_tags)
    crf_params = tf.get_variable('crf', shape = [num_tags, num_tags], dtype = tf.float32)
    pred_ids, _ = tf.contrib.crf.crf_decode(logits, crf_params, nwords) # pred_ids =  A [batch_size, max_seq_len] matrix, with dtype tf.int32.

    # Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        reverse_vocab_tags = tf.contrib.lookup.index_to_string_table_from_file(params['vocab_tags_file'])
        pred_strings = reverse_vocab_tags.lookup(tf.to_int64(pred_ids))
        predictions = {'pred_ids': pred_ids, 'tags': pred_strings}
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Loss
    vocab_tags = tf.contrib.lookup.index_table_from_file(params['vocab_tags_file'])
    label_ids = vocab_tags.lookup(labels)

    """
    logits are the same thing as unary potentials,
    checkout https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html look for scores s[i]
    """
    log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(logits, label_ids, nwords, crf_params)
    loss = tf.reduce_mean(-log_likelihood)

    # metrics
    weights = tf.sequence_mask(nwords)

    metrics = {
        'acc': tf.metrics.accuracy(label_ids, pred_ids, weights),
        'precision': precision(label_ids, pred_ids, num_tags, indices, weights), # indices indicate non-null classes
        'recall': recall(label_ids, pred_ids, num_tags, indices, weights),
        'f1': f1(label_ids, pred_ids, num_tags, indices, weights),
    }

    for metric_name, op in metrics.items():
        tf.summary.scalar(metric_name, op[1])


    # Evaluation Mode or training mode
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss = loss, eval_metric_ops = metrics )
    elif mode == tf.estimator.ModeKeys.TRAIN:
        train_op = tf.train.AdamOptimizer().minimize(loss, global_step=tf.train.get_or_create_global_step())
        return tf.estimator.EstimatorSpec(mode, loss = loss, train_op = train_op)


def fwords(name):
    return str(Path(DATADIR, '{}.words.txt'.format(name)))


def ftags(name):
    return str(Path(DATADIR, '{}.tags.txt'.format(name)))

# Write predictions to file
def write_predictions(name, estimator):
    Path('results/score').mkdir(parents=True, exist_ok=True)
    with Path('results/score/{}.preds.txt'.format(name)).open('wb') as f:
        test_inpf = functools.partial(input_fn, fwords(name), ftags(name))
        golds_gen = generator_fn(fwords(name), ftags(name))
        preds_gen = estimator.predict(test_inpf)
        for golds, preds in zip(golds_gen, preds_gen):
            ((words, _), tags) = golds
            for word, tag, tag_pred in zip(words, tags, preds['tags']):
                f.write(b' '.join([word, tag, tag_pred]) + b'\n')
            f.write(b'\n')

if __name__ == '__main__':
    # Params
    params = {
        'dim': 300,
        'dropout': 0.5,
        'num_oov_buckets': 1,
        'epochs': 25,
        'batch_size': 20,
        'buffer': 15000,
        'lstm_size': 100,
        'vocab_words_file': str(Path(DATADIR, 'vocab.words.txt')),
        'vocab_chars_file': str(Path(DATADIR, 'vocab.chars.txt')),
        'vocab_tags_file': str(Path(DATADIR, 'vocab.tags.txt')),
        'glove': str(Path(DATADIR, 'glove.npz'))
    }

    with Path('results/params.json').open('w') as f:
        json.dump(params, f, indent=4, sort_keys=True)

    print('Done writing params to disk')

    # Run configuration and estimator
    cfg = tf.estimator.RunConfig(save_checkpoints_secs=120)
    estimator = tf.estimator.Estimator(model_fn, 'results/model', cfg, params)

    print('Done creating estimator spec')

    # Defining our input functions
    train_inpf = functools.partial(input_fn, fwords('train'), ftags('train'), params, shuffle_and_repeat=True)
    eval_inpf = functools.partial(input_fn, fwords('testa'), ftags('testa'))

    # Create an early stopping hook
    Path(estimator.eval_dir()).mkdir(parents=True, exist_ok=True)

    """
    Ref: https://stackoverflow.com/questions/47137061/early-stopping-with-tf-estimator-how
    
    The parameters for stop_if_no_decrease_hook are as follows:
    
    tf.contrib.estimator.stop_if_no_decrease_hook(
    estimator,
    metric_name='loss',
    max_steps_without_decrease=1000,
    min_steps=100)
    """

    hook = tf.contrib.estimator.stop_if_no_increase_hook(estimator, 'f1', 500, min_steps=8000, run_every_secs=120)

    train_spec = tf.estimator.TrainSpec(input_fn = train_inpf, hooks = [hook])
    eval_spec = tf.estimator.EvalSpec(input_fn = eval_inpf, throttle_secs = 120) # Evaluate every 120 seconds

    print('Done creating train and eval spec')

    # Train with early stopping
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    print('Done training and evaluation')

    for name in ['train', 'testa', 'testb']:
        write_predictions(name, estimator)








