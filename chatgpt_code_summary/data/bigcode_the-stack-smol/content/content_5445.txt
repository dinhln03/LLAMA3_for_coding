import csv
import numpy as np
import re
import itertools

from collections import Counter
from collections import namedtuple


DataPoint = namedtuple('DataPoint', ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment'])


def load_datapoints(data_file):
    datapoints = []
    with open(data_file) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if 'Sentiment' not in row:
                row['Sentiment'] = None
            dp = DataPoint(**row)
            datapoints.append(dp)
    return datapoints


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def extract_phrases_in_datapoints(datapoints):
    x_text = [dp.Phrase for dp in datapoints]
    return [clean_str(sent) for sent in x_text]


def extract_phraseids_in_datapoints(datapoints):
    return [dp.PhraseId for dp in datapoints]


def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    datapoints = load_datapoints(data_file)
    x_text = extract_phrases_in_datapoints(datapoints)

    y = [int(dp.Sentiment) for dp in datapoints]

    def one_hot(i):
        return [0] * i + [1] + [0] * (4-i)

    y_vector = []
    for sentiment in y:
        y_vector.append(one_hot(sentiment))

    return [x_text, np.array(y_vector)]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
