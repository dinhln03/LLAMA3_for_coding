import copy

import torch
from torch import nn
import numpy as np

from tokens import *


def tokenize(corpus, callback=lambda sent: sent.split()):
    return [callback(sent) for sent in corpus]


def add_start_stop_tokens(corpus):
    return [[START_TOKEN] + sent + [STOP_TOKEN] for sent in corpus]


def padding(corpus, seq_len):
    for sent in corpus:
        while len(sent) < seq_len:
            sent.append(PAD_TOKEN)
        while len(sent) > seq_len:
            sent.pop()
    return corpus


def build_vocab(corpus):
    vocab = set()
    for sent in corpus:
        vocab.update(set(sent))
    vocab = list(vocab) + [UNK_TOKEN]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for idx, word in enumerate(vocab)}
    return vocab, word2idx, idx2word


def convert_to_idx(corpus, word2idx):
    return [[word2idx.get(word, "<UNK>") for word in sent] for sent in corpus]


# Output Processing
def process_output_corpus(input_seqs, preds, trues):
    new_seqs = []
    new_preds = []
    new_trues = []
    for i in range(len(input_seqs)):
        new_seq, new_pred, new_true = remove_special_tokens(
            input_seqs[i], preds[i], trues[i]
        )
        new_seqs.append(new_seq)
        new_preds.append(new_pred)
        new_trues.append(new_true)
    return new_seqs, new_preds, new_trues


def remove_special_tokens(input_seq, pred, true):
    new_seq = []
    new_pred = []
    new_true = []
    new_seq = input_seq[1:-1]
    new_true = true[1:-1]
    new_pred = pred[1:]

    # if is truncated padding
    while len(new_pred) < len(new_seq):
        new_pred.append(PAD_TOKEN)

    # if is expanded padding
    while len(new_pred) > len(new_seq):
        new_pred = new_pred[:-1]

    return new_seq, new_pred, new_true


def convert_to_token(corpus, idx2token):
    return [[idx2token[token_idx] for token_idx in sent] for sent in corpus]


def preprocess_utterances(utterances, utterance_dataset):
    # tokenization
    utterances = tokenize(utterances)
    # add special tokens
    utterances = add_start_stop_tokens(utterances)
    tokenized_utterances = copy.deepcopy(utterances)
    # padding
    utterances = padding(utterances, utterance_dataset.seq_len)

    word2idx = utterance_dataset.word2idx
    utterances = [
        [word2idx.get(token, word2idx[UNK_TOKEN]) for token in sent]
        for sent in utterances
    ]

    return utterances, tokenized_utterances


def read_glove_vector(glove_vec):
    with open(glove_vec, "r", encoding="UTF-8") as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            w_line = line.split()
            curr_word = w_line[0]
            word_to_vec_map[curr_word] = np.array(w_line[1:], dtype=np.float64)

    return word_to_vec_map


# functions for creating the embedding layer
def get_one_hot_matrix(vocab):
    one_hot_matrix = np.zeros((len(vocab), len(vocab)))
    np.fill_diagonal(one_hot_matrix, 1)
    return one_hot_matrix


def get_glove_matrix(glove_map, vocab):
    matrix_len = len(vocab)
    emb_dim = len(list(glove_map.values())[0])
    weights_matrix = np.zeros((matrix_len, emb_dim))
    for i, word in enumerate(vocab):
        try:
            weights_matrix[i] = glove_map[word]
        except KeyError:
            if word in [PAD_TOKEN, START_TOKEN, STOP_TOKEN]:
                weights_matrix[i] = np.zeros((emb_dim,))
            else:
                weights_matrix[i] = np.random.normal(
                    scale=0.6, size=(emb_dim,)
                )
    return weights_matrix


def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({"weight": torch.tensor(weights_matrix)})
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer, num_embeddings, embedding_dim

