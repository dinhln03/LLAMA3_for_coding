#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Zimeng Qiu <zimengq@andrew.cmu.edu>

"""
F19 11-411/611 NLP Assignment 3 Task 1
N-gram Language Model Implementation Script
Zimeng Qiu Sep 2019

This is a simple implementation of N-gram language model

Write your own implementation in this file!
"""

import argparse
from utils import *
import numpy as np


class LanguageModel(object):
    """
    Base class for all language models
    """
    def __init__(self, corpus, ngram, min_freq, uniform=False):
        """
        Initialize language model
        :param corpus: input text corpus to build LM on
        :param ngram: number of n-gram, e.g. 1, 2, 3, ...
        :param min_freq: minimum frequency threshold to set a word to UNK placeholder
                         set to 1 to not use this threshold
        :param uniform: boolean flag, set to True to indicate this model is a simple uniform LM
                        otherwise will be an N-gram model
        """
        # write your initialize code below
        self.corpus = corpus
        self.ngram = ngram
        self.min_freq = min_freq
        self.uniform = uniform

        self.uniform_table = None
        self.unigram_table = None
        self.bigram_table = None
        self.trigram_table = None

        self.infrequent_words = find_infrequent_words(self.corpus,self.min_freq)
        replace_infrequent_words(self.corpus,self.infrequent_words)

        self.corpus_1gram,self.vocabulary,self.V,self.N = get_vocabulary(self.corpus)
        self.word_to_idx,self.idx_to_word = get_word_mappings(self.vocabulary)
        self.counter_1gram = get_counter(self.corpus_1gram)

        self.build()

    def build(self):
        """
        Build LM from text corpus
        """
        # Write your own implementation here
        
        # uniform
        if self.uniform:
            self.uniform_table = get_uniform_tables(self.V)
        else:
            # unigram
            if self.ngram == 1:
                self.unigram_table = get_unigram_tables(self.V,self.N,self.counter_1gram,self.word_to_idx)
            # bigram
            elif self.ngram == 2:
                self.corpus_2gram = [(self.corpus_1gram[i],self.corpus_1gram[i+1]) for i in range(len(self.corpus_1gram)-1)]
                self.counter_2gram = get_counter(self.corpus_2gram)

                self.bigram_table = get_bigram_tables(self.V,self.counter_1gram,self.counter_2gram,self.word_to_idx,self.idx_to_word)
            # trigram
            elif self.ngram == 3:
                self.corpus_2gram = [(self.corpus_1gram[i],self.corpus_1gram[i+1]) for i in range(len(self.corpus_1gram)-1)]
                self.counter_2gram = get_counter(self.corpus_2gram)

                self.corpus_3gram = [(self.corpus_1gram[i],self.corpus_1gram[i+1],self.corpus_1gram[i+2]) for i in range(len(self.corpus_1gram)-2)]
                self.counter_3gram = get_counter(self.corpus_3gram)

                self.trigram_table = get_trigram_tables(self.V,self.counter_2gram,self.counter_3gram,self.word_to_idx)

    def most_common_words(self, k):
        """
        Return the top-k most frequent n-grams and their frequencies in sorted order.
        For uniform models, the frequency should be "1" for each token.

        Your return should be sorted in descending order of frequency.
        Sort according to ascending alphabet order when multiple words have same frequency.
        :return: list[tuple(token, freq)] of top k most common tokens
        """
        # Write your own implementation here

        if self.uniform:
            return [(word,1) for word in sorted(self.vocabulary)[0:k]]
        else:
            if self.ngram == 1:
                return sorted(self.counter_1gram.most_common(),key=lambda x:(-x[1],x[0]))[0:k]
            elif self.ngram == 2:
                return [(token[0]+' '+token[1],num) for token, num in sorted(self.counter_2gram.most_common(),key=lambda x:(-x[1],x[0]))[0:k]]
            elif self.ngram == 3:
                return [(token[0]+' '+token[1]+' '+token[2],num) for token,num in sorted(self.counter_3gram.most_common(),key=lambda x:(-x[1],x[0]))[0:k]]
        return


def calculate_perplexity(models, coefs, data):
    """
    Calculate perplexity with given model
    :param models: language models
    :param coefs: coefficients
    :param data: test data
    :return: perplexity
    """
    # Write your own implementation here
    pp = 0
    uniform_prob = []
    unigram_prob = []
    bigram_prob = []
    trigram_prob = []
    
    prob_table_unifrom = None
    prob_table_1gram = None
    prob_table_2gram = None
    prob_table_3gram = None


    min_freq = models[0].min_freq
    train_vocabulary = models[0].vocabulary
    word_to_idx,idx_to_word = models[0].word_to_idx,models[0].idx_to_word

    test_infrequent_words = find_infrequent_words(data,min_freq)
    replace_infrequent_words(data,test_infrequent_words)

    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] not in train_vocabulary:
                data[i][j] = 'UNK'
    
    corpus_1gram,vocabulary,V,N = get_vocabulary(data)
    corpus_2gram = [(corpus_1gram[i],corpus_1gram[i+1]) for i in range(len(corpus_1gram)-1)]
    corpus_3gram = [(corpus_1gram[i],corpus_1gram[i+1],corpus_1gram[i+2]) for i in range(len(corpus_1gram)-2)]

    for i in range(len(models)):
        model = models[i]
        if model.uniform:
            prob_table_unifrom = model.uniform_table
            for word in corpus_1gram:
                uniform_prob.append(prob_table_unifrom[0][word_to_idx[word]]*coefs[0])
        else:
            if model.ngram == 1:
                prob_table_1gram = model.unigram_table
                for word in corpus_1gram:
                    unigram_prob.append(prob_table_1gram[0][word_to_idx[word]]*coefs[1])
            elif model.ngram == 2:
                prob_table_2gram = model.bigram_table
                bigram_prob.append(prob_table_1gram[0][word_to_idx[corpus_2gram[0][0]]])
                for words in corpus_2gram:
                    word1 = words[0]
                    word2 = words[1]

                    prob_1gram = prob_table_1gram[0][word_to_idx[word2]]
                    prob_2gram = prob_table_2gram[word_to_idx[word1]][word_to_idx[word2]]

                    if prob_2gram != 0:
                        bigram_prob.append(prob_2gram*coefs[2])
                    else:
                        bigram_prob.append(prob_1gram*coefs[2])

            elif model.ngram == 3:
                prob_table_3gram = model.trigram_table
                train_corpus_3gram = set(model.corpus_3gram)

                trigram_prob.append(prob_table_1gram[0][word_to_idx[corpus_3gram[0][0]]])
                trigram_prob.append(prob_table_1gram[0][word_to_idx[corpus_3gram[0][1]]])
                for words in corpus_3gram:
                    word1 = words[0]
                    word2 = words[1]
                    word3 = words[2]
                    if words in train_corpus_3gram:
                        prob_3gram = prob_table_3gram[(word1,word2,word3)]
                        trigram_prob.append(prob_3gram*coefs[3])
                    else:
                        prob_1gram = prob_table_1gram[0][word_to_idx[word3]]
                        prob_2gram = prob_table_2gram[word_to_idx[word2]][word_to_idx[word3]]
                        if prob_2gram != 0:
                            trigram_prob.append(prob_2gram*coefs[3])
                        else:
                            trigram_prob.append(prob_1gram*coefs[3])


    prob = np.zeros((N,),dtype=np.float64)
    for i in range(len(prob)):
        prob[i] += uniform_prob[i]
        prob[i] += unigram_prob[i]
        prob[i] += bigram_prob[i]
        prob[i] += trigram_prob[i]

    for p in prob:
        pp += np.log2(p)
    
    pp /= -N
    pp = np.power(2,pp)

    return pp

# Do not modify this function!
def parse_args():
    """
    Parse input positional arguments from command line
    :return: args - parsed arguments
    """
    parser = argparse.ArgumentParser('N-gram Language Model')
    parser.add_argument('coef_unif', help='coefficient for the uniform model.', type=float)
    parser.add_argument('coef_uni', help='coefficient for the unigram model.', type=float)
    parser.add_argument('coef_bi', help='coefficient for the bigram model.', type=float)
    parser.add_argument('coef_tri', help='coefficient for the trigram model.', type=float)
    parser.add_argument('min_freq', type=int,
                        help='minimum frequency threshold for substitute '
                             'with UNK token, set to 1 for not use this threshold')
    parser.add_argument('testfile', help='test text file.')
    parser.add_argument('trainfile', help='training text file.', nargs='+')
    return parser.parse_args()


# Main executable script provided for your convenience
# Not executed on autograder, so do what you want
if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # load and preprocess train and test data
    train = preprocess(load_dataset(args.trainfile))
    test = preprocess(read_file(args.testfile))

    # build language models
    uniform = LanguageModel(train, ngram=1, min_freq=args.min_freq, uniform=True)
    unigram = LanguageModel(train, ngram=1, min_freq=args.min_freq)
    # print('Unique 1-gram types:',len(unigram.counter_1gram.most_common()))
    # print('top 15 unigram:',unigram.counter_1gram.most_common()[:15])
    bigram = LanguageModel(train, ngram=2, min_freq=args.min_freq)
    # print('Unique 2-gram types:',len(bigram.counter_2gram.most_common()))
    # print('top 15 bigram:',bigram.counter_2gram.most_common()[:15])
    trigram = LanguageModel(train, ngram=3, min_freq=args.min_freq)
    # print('Unique 3-gram types:',len(trigram.counter_3gram.most_common()))
    # print('top 15 trigram:',trigram.counter_3gram.most_common()[:50])

    # calculate perplexity on test file
    ppl = calculate_perplexity(
        models=[uniform, unigram, bigram, trigram],
        coefs=[args.coef_unif, args.coef_uni, args.coef_bi, args.coef_tri],
        data=test)

    print("Perplexity: {}".format(ppl))
