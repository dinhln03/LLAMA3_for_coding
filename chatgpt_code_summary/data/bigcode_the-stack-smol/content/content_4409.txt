# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2019-02-13 12:41:44

from __future__ import print_function
import time
import sys
import argparse
import random
import torch
import gc
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.metric import get_ner_fmeasure
from model.seqlabel import SeqLabel
from model.sentclassifier import SentClassifier
from utils.data import Data

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap



try:
    import cPickle as pickle
except ImportError:
    import pickle

DEFAULT_TRAINED_FILE = 'test_data/lstmtestglove50.9.model'

seed_num = 46
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

def importance_matrix(sensitivities, data,
                      print_imp=True, show_table=True, tag_to_ablate=None):
    '''
    Builds a matrix of tag sensitivities
    :param sensitivities: This is a matrix of [num_tags, num_neurons],
    which is [10 x 50] in our experimental configuration.
    :return:
    '''

    important_lists = []
    important_nps = np.zeros(50, dtype=int)
    sensitivities = sensitivities[1:]  # omit padding tag
    for i in range(len(sensitivities)):
        important_list = []
        important_np = np.zeros(50, dtype=int)
        tag_sensitivity_row = sensitivities[i]
        for j in range(len(tag_sensitivity_row)):
            most_important = np.argmax(tag_sensitivity_row)
            important_list.append(most_important)
            important_np[j] = most_important
            index = [most_important]
            tag_sensitivity_row[most_important] = np.NINF
        important_lists.append(important_list)
        important_nps = np.vstack((important_nps, important_np))

    important_nps = np.delete(important_nps, 0, axis=0) # delete padding tag
    np.save("imps.npy",important_nps) # save importance rows for other scripts to use

    important_nps = np.transpose(important_nps)
    if show_table:
        sns.set()
        # Smaller than normal fonts
        sns.set(font_scale=0.5)
        x_tick = [data.label_alphabet.get_instance(tag) for tag in sorted(data.tag_counts)]
        del(x_tick[0])
        ax = sns.heatmap(important_nps, annot=True, xticklabels=x_tick,
                         cmap=ListedColormap(['white']), cbar=False, yticklabels=False,
                         linecolor='gray', linewidths=0.4)
        title = "Importance rankings of neurons per tag"
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])
        plt.show()

        def trim_model_dir(model_dir):
            model_dir = model_dir.replace('/','-')
            return model_dir
        ax.figure.savefig("ImportanceRankings-{}.png".format(trim_model_dir(data.model_dir)))
    if print_imp:
        imp_file = open("Importance-{}.txt".format(trim_model_dir(data.model_dir)), "w+")
        print('Neuron importance ranking for each NER tag:')
        for i, l in enumerate(important_lists):
            tags = [data.label_alphabet.get_instance(tag) for tag in sorted(data.tag_counts)]
            del(tags[0]) # remove PAD tag
            print ("\t{}\t{}".format(tags[i], l))
            imp_file.write("{}\t{}\n".format(tags[i], l))
        imp_file.write("\n")
        np.savetxt("Importance-{}.tsv".format(trim_model_dir(data.model_dir)),
                   important_nps, fmt='%2.0d', delimiter='\t')

    return important_nps

def heatmap_sensitivity(sensitivities,
                        modelname=DEFAULT_TRAINED_FILE,
                        testname="",
                        show_pad=False,
                        show_vals=True,
                        disable=False):
    '''
    Shows a heatmap for the sensitivity values, saves the heatmap to a PNG file,
    and also saves the sensitivity matrix to an .npy file,
    which we use for calculating correlations between models later.
    :param sensitivities: This is a matrix of [num_tags, num_neurons],
    which is [10 x 50] in our experimental configuration.
    :param disable: disable is just to turn off for debugging
    :return:
    '''
    # transpose to match chart in Figure 7. of paper
    sensitivities = np.transpose(sensitivities)
    # column 0 is the padding tag
    start = 1
    if show_pad:
        start = 0
    sensitivities = sensitivities[0:50, start:10]
    sns.set()
    # Smaller than normal fonts
    sns.set(font_scale=0.5)
    x_tick = [data.label_alphabet.get_instance(tag) for tag in sorted(data.tag_counts)]
    if show_pad: x_tick[0] = 'PAD'
    else: del(x_tick[0])

    # change tags' order to use in downstream correlation diagrams
    sensitivities_temp = np.zeros((50, 9))
    x_tick_output = ['B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-MISC', 'I-MISC', 'O']
    for i in range(len(x_tick_output)):
        sensitivities_temp[:, i] = sensitivities[:, x_tick.index(x_tick_output[i])]
    np.save(modelname+'_sensitivities.npy', sensitivities_temp)
    
    # put sensititivites in heat map
    if not disable:
        ax = sns.heatmap(sensitivities, xticklabels=x_tick, annot=show_vals, fmt=".2g")
        title = "({}): ".format(testname) + modelname
        plt.title(title, fontsize=18)
        ttl = ax.title
        ttl.set_position([0.5, 1.05])
        plt.show()
        ax.figure.savefig(modelname+"_heatmap.png")


def get_sensitivity_matrix(label, debug=True):
    '''
    Given a tag like 4: (B-PER), return the sensitivity matrix
    :param label:
    :return:
    '''

    avg_for_label = data.tag_contributions[label]/data.tag_counts[label]
    sum_other_counts = 0

    # data.tag_contributions[0] is for the padding label and can be ignored
    sum_other_contributions = np.zeros((10, 50))
    for l in data.tag_counts:

        if l != label and l != 0:  #  if l != label: (to consider the padding label which is 0)
            sum_other_counts += data.tag_counts[l]
            sum_other_contributions += data.tag_contributions[l]
    avg_for_others = sum_other_contributions/sum_other_counts

    s_ij = avg_for_label - avg_for_others
    s_ij_label = s_ij[label]
    return s_ij_label  # was return s_ij


def data_initialization(data):
    data.initial_feature_alphabets()
    data.build_alphabet(data.train_dir)
    data.build_alphabet(data.dev_dir)
    data.build_alphabet(data.test_dir)
    data.fix_alphabet()


def predict_check(pred_variable, gold_variable, mask_variable, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result, in numpy format
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred = pred_variable.cpu().data.numpy()
    gold = gold_variable.cpu().data.numpy()
    mask = mask_variable.cpu().data.numpy()
    overlaped = (pred == gold)
    if sentence_classification:
        # print(overlaped)
        # print(overlaped*pred)
        right_token = np.sum(overlaped)
        total_token = overlaped.shape[0] ## =batch_size
    else:
        right_token = np.sum(overlaped * mask)
        total_token = mask.sum()
    # print("right: %s, total: %s"%(right_token, total_token))
    return right_token, total_token


def recover_label(pred_variable, gold_variable, mask_variable, label_alphabet, word_recover, sentence_classification=False):
    """
        input:
            pred_variable (batch_size, sent_len): pred tag result
            gold_variable (batch_size, sent_len): gold result variable
            mask_variable (batch_size, sent_len): mask variable
    """
    pred_variable = pred_variable[word_recover]
    # print("reordered labels: {}".format(pred_variable))
    gold_variable = gold_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = gold_variable.size(0)
    if sentence_classification:
        pred_tag = pred_variable.cpu().data.numpy().tolist()
        gold_tag = gold_variable.cpu().data.numpy().tolist()
        pred_label = [label_alphabet.get_instance(pred) for pred in pred_tag]
        gold_label = [label_alphabet.get_instance(gold) for gold in gold_tag]
    else:
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert(len(pred)==len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
    return pred_label, gold_label


def recover_nbest_label(pred_variable, mask_variable, label_alphabet, word_recover):
    """
        input:
            pred_variable (batch_size, sent_len, nbest): pred tag result
            mask_variable (batch_size, sent_len): mask variable
            word_recover (batch_size)
        output:
            nbest_pred_label list: [batch_size, nbest, each_seq_len]
    """
    # exit(0)
    pred_variable = pred_variable[word_recover]
    mask_variable = mask_variable[word_recover]
    batch_size = pred_variable.size(0)
    seq_len = pred_variable.size(1)
    nbest = pred_variable.size(2)
    mask = mask_variable.cpu().data.numpy()
    pred_tag = pred_variable.cpu().data.numpy()
    batch_size = mask.shape[0]
    pred_label = []
    for idx in range(batch_size):
        pred = []
        for idz in range(nbest):
            each_pred = [label_alphabet.get_instance(pred_tag[idx][idy][idz]) for idy in range(seq_len) if mask[idx][idy] != 0]
            pred.append(each_pred)
        pred_label.append(pred)
    return pred_label


def lr_decay(optimizer, epoch, decay_rate, init_lr):
    lr = init_lr/(1+decay_rate*epoch)
    print(" Learning rate is set as:", lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(data, model, name, nbest=None, print_tag_counts=False, tag_to_ablate=None):
    '''

    :param data:
    :param model:
    :param name:
    :param nbest:
    :param print_tag_counts:
    :param tag_to_ablate: if this is set to a tag name, like 'B-ORG', then in the LSTM layer's forward() we ablate the
    number of neurons specified by data.ablate_num
    :return:
    '''
    ablate_list_for_tag = None
    if tag_to_ablate:
        data.ablate_tag = tag_to_ablate
        ablate_list_for_tag = data.ablate_list[tag_to_ablate]

    print("\nEVALUATE file: {}, set={}, \n\t ablate_num={} tag: {} \nablate_list_for_tag={}".format(
        data.model_dir, name, data.current_ablate_ind, tag_to_ablate, ablate_list_for_tag))
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
        exit(1)
    right_token = 0
    whole_token = 0
    nbest_pred_results = []
    pred_scores = []
    pred_results = []
    gold_results = []
    ## set model in eval model
    model.eval()

    ''' Get count of model parameters '''
    # print("COUNT PARAMETERS: {}".format(count_parameters(model)))

    batch_size = data.HP_batch_size
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num//batch_size+1
    for batch_id in range(total_batch):
        start = batch_id*batch_size
        end = (batch_id+1)*batch_size
        if end > train_num:
            end =  train_num
        instance = instances[start:end]
        if not instance:
            continue
        batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, False, data.sentence_classification)
        if nbest and not data.sentence_classification:
            scores, nbest_tag_seq = model.decode_nbest(batch_word,batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask, nbest)
            nbest_pred_result = recover_nbest_label(nbest_tag_seq, mask, data.label_alphabet, batch_wordrecover)
            nbest_pred_results += nbest_pred_result
            pred_scores += scores[batch_wordrecover].cpu().data.numpy().tolist()
            ## select the best sequence to evalurate
            tag_seq = nbest_tag_seq[:,:,0]
        else:
            tag_seq = model(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)




        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover, data.sentence_classification)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances)/decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme, data=data)
    if nbest and not data.sentence_classification:
        return speed, acc, p, r, f, nbest_pred_results, pred_scores

    ''' Get per-tag sensitivity '''
    ## print("TOTAL BATCH ITERATIONS: {}".format(data.iteration))
    sensitivity_matrices = []  # This will hold a row for each tag's sensitivity
    for tag in sorted(data.tag_counts):
        if print_tag_counts:
            if tag == 0:
                print("Padding {}: {} instances.".format('0', data.tag_counts[tag]))
            else:
                print("Tag {}: {} instances.".format(data.label_alphabet.get_instance(tag), data.tag_counts[tag]))
        sensitivity_tag = get_sensitivity_matrix(tag)
        sensitivity_matrices.append(sensitivity_tag)

    sensitivity_combined = np.squeeze(np.stack([sensitivity_matrices]))
    # TODO: the following line would stack multiple models' sensitivity,
    # but we don't need it unless running many different models for stats
    # data.sensitivity_matrices_combined.append(sensitivity_combined)
    return speed, acc, p, r, f, pred_results, pred_scores, sensitivity_combined



def batchify_with_label(input_batch_list, gpu, if_train=True, sentence_classification=False):
    if sentence_classification:
        return batchify_sentence_classification_with_label(input_batch_list, gpu, if_train)
    else:
        return batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train)


def batchify_sequence_labeling_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, sent_len, feature_num)
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size, sent_len)

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size, max_sent_len),...] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, max_sent_len)
            mask: (batch_size, max_sent_len)
    """
    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]
    feature_num = len(features[0][0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())
    # '
    ''' 517 '''
    # mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad=if_train).bool()
    for idx, (seq, label, seqlen) in enumerate(zip(words, labels, word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        label_seq_tensor[idx, :seqlen] = torch.LongTensor(label)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]

    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def batchify_sentence_classification_with_label(input_batch_list, gpu, if_train=True):
    """
        input: list of words, chars and labels, various length. [[words, features, chars, labels],[words, features, chars,labels],...]
            words: word ids for one sentence. (batch_size, sent_len)
            features: features ids for one sentence. (batch_size, feature_num), each sentence has one set of feature
            chars: char ids for on sentences, various length. (batch_size, sent_len, each_word_length)
            labels: label ids for one sentence. (batch_size,), each sentence has one set of feature

        output:
            zero padding for word and char, with their batch length
            word_seq_tensor: (batch_size, max_sent_len) Variable
            feature_seq_tensors: [(batch_size,), ... ] list of Variable
            word_seq_lengths: (batch_size,1) Tensor
            char_seq_tensor: (batch_size*max_sent_len, max_word_len) Variable
            char_seq_lengths: (batch_size*max_sent_len,1) Tensor
            char_seq_recover: (batch_size*max_sent_len,1)  recover char sequence order
            label_seq_tensor: (batch_size, )
            mask: (batch_size, max_sent_len)
    """

    batch_size = len(input_batch_list)
    words = [sent[0] for sent in input_batch_list]
    features = [np.asarray(sent[1]) for sent in input_batch_list]    
    feature_num = len(features[0])
    chars = [sent[2] for sent in input_batch_list]
    labels = [sent[3] for sent in input_batch_list]
    word_seq_lengths = torch.LongTensor(list(map(len, words)))
    max_seq_len = word_seq_lengths.max().item()
    word_seq_tensor = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).long()
    label_seq_tensor = torch.zeros((batch_size, ), requires_grad =  if_train).long()
    feature_seq_tensors = []
    for idx in range(feature_num):
        feature_seq_tensors.append(torch.zeros((batch_size, max_seq_len),requires_grad =  if_train).long())

    ''' 517 '''
    # mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).byte()
    mask = torch.zeros((batch_size, max_seq_len), requires_grad =  if_train).bool()
    label_seq_tensor = torch.LongTensor(labels)
    # exit(0)
    for idx, (seq,  seqlen) in enumerate(zip(words,  word_seq_lengths)):
        seqlen = seqlen.item()
        word_seq_tensor[idx, :seqlen] = torch.LongTensor(seq)
        mask[idx, :seqlen] = torch.Tensor([1]*seqlen)
        for idy in range(feature_num):
            feature_seq_tensors[idy][idx,:seqlen] = torch.LongTensor(features[idx][:,idy])
    word_seq_lengths, word_perm_idx = word_seq_lengths.sort(0, descending=True)
    word_seq_tensor = word_seq_tensor[word_perm_idx]
    for idx in range(feature_num):
        feature_seq_tensors[idx] = feature_seq_tensors[idx][word_perm_idx]
    label_seq_tensor = label_seq_tensor[word_perm_idx]
    mask = mask[word_perm_idx]
    ### deal with char
    # pad_chars (batch_size, max_seq_len)
    pad_chars = [chars[idx] + [[0]] * (max_seq_len-len(chars[idx])) for idx in range(len(chars))]
    length_list = [list(map(len, pad_char)) for pad_char in pad_chars]
    max_word_len = max(map(max, length_list))
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_word_len), requires_grad =  if_train).long()
    char_seq_lengths = torch.LongTensor(length_list)
    for idx, (seq, seqlen) in enumerate(zip(pad_chars, char_seq_lengths)):
        for idy, (word, wordlen) in enumerate(zip(seq, seqlen)):
            # print len(word), wordlen
            char_seq_tensor[idx, idy, :wordlen] = torch.LongTensor(word)

    char_seq_tensor = char_seq_tensor[word_perm_idx].view(batch_size*max_seq_len,-1)
    char_seq_lengths = char_seq_lengths[word_perm_idx].view(batch_size*max_seq_len,)
    char_seq_lengths, char_perm_idx = char_seq_lengths.sort(0, descending=True)
    char_seq_tensor = char_seq_tensor[char_perm_idx]
    _, char_seq_recover = char_perm_idx.sort(0, descending=False)
    _, word_seq_recover = word_perm_idx.sort(0, descending=False)
    if gpu:
        word_seq_tensor = word_seq_tensor.cuda()
        for idx in range(feature_num):
            feature_seq_tensors[idx] = feature_seq_tensors[idx].cuda()
        word_seq_lengths = word_seq_lengths.cuda()
        word_seq_recover = word_seq_recover.cuda()
        label_seq_tensor = label_seq_tensor.cuda()
        char_seq_tensor = char_seq_tensor.cuda()
        char_seq_recover = char_seq_recover.cuda()
        mask = mask.cuda()
    return word_seq_tensor,feature_seq_tensors, word_seq_lengths, word_seq_recover, char_seq_tensor, char_seq_lengths, char_seq_recover, label_seq_tensor, mask


def load_model_to_test(data, train=False, dev=True, test=False, tag=None):
    '''
    Set any ONE of train, dev, test to true, in order to evaluate on that set.
    :param data:
    :param train:
    :param dev: Default set to test, because that was what the original experiment did
    :param test:
    :return:
    '''

    print("Load pretrained model...")
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    model.load_state_dict(torch.load(data.pretrained_model_path))



    '''----------------TESTING----------------'''
    if (train):
        speed, acc, p, r, f, _,_, train_sensitivities = evaluate(data, model, "train")
        heatmap_sensitivity(train_sensitivities, data.pretrained_model_path, testname="train")
        if data.seg:
            current_score = f
            print("Speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(speed, acc, p, r, f))
        else:
            current_score = acc
            print("Speed: %.2fst/s; acc: %.4f"%(speed, acc))

    if (dev):
        # for tag in data.ablate_list:
        speed, acc, p, r, f, _,_, sensitivities = evaluate(
            data, model, "dev", tag_to_ablate=tag)
        if data.seg:
            current_score = f
            print("Speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (speed, acc, p, r, f))
        else:
            current_score = acc
            print("Speed: %.2fst/s; acc: %.4f" % (speed, acc))

        if (data.ablate_num == 0):
            heatmap_sensitivity(sensitivities, data.pretrained_model_path, testname="dev")
            importance_matrix(sensitivities, data)




    if (test):
        speed, acc, p, r, f, _,_ = evaluate(data, model, "test")
        if data.seg:
            print("Speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(speed, acc, p, r, f))
        else:
            print("Speed: %.2fst/s; acc: %.4f"%(speed, acc))

    return


def train(data):
    print("Training model...")
    data.show_data_summary()
    save_data_name = data.model_dir +".dset"
    data.save(save_data_name)
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)

    if data.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=data.HP_lr, momentum=data.HP_momentum,weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adadelta":
        optimizer = optim.Adadelta(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    elif data.optimizer.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=data.HP_lr, weight_decay=data.HP_l2)
    else:
        print("Optimizer illegal: %s"%(data.optimizer))
        exit(1)
    best_dev = -10
    # data.HP_iteration = 1
    ## start training
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" %(idx,data.HP_iteration))
        if data.optimizer == "SGD":
            optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        instance_count = 0
        sample_id = 0
        sample_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random.shuffle(data.train_Ids)
        print("Shuffle: first input word list:", data.train_Ids[0][0])
        ## set model in train model
        model.train()
        model.zero_grad()
        batch_size = data.HP_batch_size
        batch_id = 0
        train_num = len(data.train_Ids)
        total_batch = train_num//batch_size+1
        for batch_id in range(total_batch):
            start = batch_id*batch_size
            end = (batch_id+1)*batch_size
            if end >train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            batch_word, batch_features, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask  = batchify_with_label(instance, data.HP_gpu, True, data.sentence_classification)
            instance_count += 1
            loss, tag_seq = model.calculate_loss(batch_word, batch_features, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask, data.sentence_classification)
            right_token += right
            whole_token += whole
            # print("loss:",loss.item())
            sample_loss += loss.item()
            total_loss += loss.item()
            if end%500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))
                if sample_loss > 1e8 or str(sample_loss) == "nan":
                    print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
                    exit(1)
                sys.stdout.flush()
                sample_loss = 0
            loss.backward()
            optimizer.step()
            model.zero_grad()
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f"%(end, temp_cost, sample_loss, right_token, whole_token,(right_token+0.)/whole_token))

        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s"%(idx, epoch_cost, train_num/epoch_cost, total_loss))
        print("totalloss:", total_loss)
        if total_loss > 1e8 or str(total_loss) == "nan":
            print("ERROR: LOSS EXPLOSION (>1e8) ! PLEASE SET PROPER PARAMETERS AND STRUCTURE! EXIT....")
            exit(1)
        # continue
        speed, acc, p, r, f, _,_ , sensitivities = evaluate(data, model, "dev")

        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish

        if data.seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f"%(dev_cost, speed, acc))

        if current_score > best_dev:
            if data.seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_name = data.model_dir +'.'+ str(idx) + ".model"
            print("Save current best model in file:", model_name)
            torch.save(model.state_dict(), model_name)
            best_dev = current_score
        # ## decode test
        speed, acc, p, r, f, _,_ , sensitivities = evaluate(data, model, "test")


        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if data.seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f"%(test_cost, speed, acc))
        gc.collect()


def load_model_decode(data, name):
    print("Load Model from file: {}, name={}".format(data.model_dir, name) )
    if data.sentence_classification:
        model = SentClassifier(data)
    else:
        model = SeqLabel(data)
    # model = SeqModel(data)
    ## load model need consider if the model trained in GPU and load in CPU, or vice versa
    # if not gpu:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model.load_state_dict(torch.load(model_dir), map_location=lambda storage, loc: storage)
    #     # model = torch.load(model_dir, map_location=lambda storage, loc: storage)
    # else:
    #     model.load_state_dict(torch.load(model_dir))
    #     # model = torch.load(model_dir)
    model.load_state_dict(torch.load(data.load_model_dir))

    print("Decode %s data, nbest: %s ..."%(name, data.nbest))
    start_time = time.time()
    speed, acc, p, r, f, pred_results, pred_scores = evaluate(data, model, name, data.nbest)
    end_time = time.time()
    time_cost = end_time - start_time
    if data.seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f"%(name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f"%(name, time_cost, speed, acc))
    return pred_results, pred_scores


def load_ablation_file():
    filename = ("Importance-" + data.model_dir + ".txt").replace('/','-')
    ablate_lists = {}
    ''' B-ORG	[4, 24, 14, 15, 19, 46, 36, 22, 27, 9, 13, 20, 25, 33, 45, 0, 35, 40, 48, 42, 44, 18, 37, 21, 32, 29, 16, 26, 11, 7, 23, 49, 12, 5, 8, 38, 2, 47, 1, 43, 31, 30, 41, 6, 28, 3, 34, 39, 10, 17]'''
    with open(filename, 'r+') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if len(line) > 0:
                (tag, list) = line.split('[')[0].strip(), line.split('[')[1].strip().replace(']','')
                list = list.split(',')
                ablate_lists[tag] = [int(i) for i in list]
    return ablate_lists

def clear_sensitivity_data():
    data.iteration = 0
    data.batch_contributions = []
    data.tag_contributions = {}
    data.tag_counts = {}
    data.sensitivity_matrices = []
    data.sensitivity_matrices_combined = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tuning with NCRF++')
    # parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--config',  help='Configuration File', default='None')
    parser.add_argument('--wordemb',  help='Embedding for words', default='None')
    parser.add_argument('--charemb',  help='Embedding for chars', default='None')
    parser.add_argument('--status', choices=['train', 'decode'], help='update algorithm', default='train')
    parser.add_argument('--savemodel', default="data/model/saved_model.lstmcrf.")
    parser.add_argument('--savedset', help='Dir of saved data setting')
    parser.add_argument('--train', default="data/conll03/train.bmes") 
    parser.add_argument('--dev', default="data/conll03/dev.bmes" )  
    parser.add_argument('--test', default="data/conll03/test.bmes") 
    parser.add_argument('--seg', default="True") 
    parser.add_argument('--raw') 
    parser.add_argument('--loadmodel')
    parser.add_argument('--output')
    parser.add_argument('--loadtotest', help='Load the model just to test it')
    parser.add_argument('--pretrainedmodelpath', help='Path to a pretrained model that you just want to test',
                        default=DEFAULT_TRAINED_FILE)
    parser.add_argument('--ablate', help='how many neurons to ablate', default=0) # indicate number of neurons to ablate
    # Importance.txt is generated by importance_matrix() (automatically reading this file is a TODO)
    parser.add_argument('--ablate_file', help='list of neurons to ablate')

    args = parser.parse_args()
    data = Data()
    data.HP_gpu = torch.cuda.is_available()
    if args.config == 'None':
        data.train_dir = args.train 
        data.dev_dir = args.dev 
        data.test_dir = args.test
        data.model_dir = args.savemodel
        data.dset_dir = args.savedset
        print("Save dset directory:",data.dset_dir)
        save_model_dir = args.savemodel
        data.word_emb_dir = args.wordemb
        data.char_emb_dir = args.charemb
        if args.seg.lower() == 'true':
            data.seg = True
        else:
            data.seg = False
        print("Seed num:",seed_num)
    else:
        data.read_config(args.config)

    # adding arg for pretrained model path
    data.pretrained_model_path = args.pretrainedmodelpath
    data.ablate_num = int(args.ablate)
    # data.show_data_summary()
    status = data.status.lower()
    print("Seed num:",seed_num)

    if status == 'train':
        print("MODEL: train")
        data_initialization(data)  # set up alphabets
        data.generate_instance('train')
        data.generate_instance('dev')
        data.generate_instance('test')
        data.build_pretrain_emb()
        if not args.loadtotest:
            print("Training model, not just testing because --loadtotest is {}".format(args.loadtotest))
            print("Loading ablation file even though it's just a placeholder")
            debug_ablation = False
            if debug_ablation:
                data.ablate_list = load_ablation_file()  # TODO: file not found
                tag_list = data.ablate_list.keys()
            train(data)
        else:
            if args.ablate:
                data.ablate_num = int(args.ablate)
            print("Loading model to test.")
            data.ablate_list = load_ablation_file()
            tag_list = data.ablate_list.keys()
            # todo: command line arg for specific current ablate index
            # todo: command line arg for intervals

            for tag in tag_list:
                data.ablate_tag = tag
                data.current_ablate_ind[tag] = 0
                data.acc_chart[data.ablate_tag] = {} # clear accuracy dict of lists for the tag
                for i in range(0, data.ablate_num + 1):
                    data.current_ablate_ind[tag] = i #+= 1 # todo: option to skip by different interval like every 5
                    clear_sensitivity_data()
                    load_model_to_test(data, tag=tag)

            # print out acc_chart
            #for tag in data.ablate_list:
                print ('{} ABLATION RESULTS:'.format(tag))
                degradations = {}
                for t in tag_list:
                    print("\tTag: {}, Decr. Accs: {}".format(t, data.acc_chart[tag][t]))
                    degradations[t] = \
                        [data.acc_chart[tag][t][ind] - data.acc_chart[tag][t][0] for ind in range (0, data.ablate_num+1)]
                    print("\t\tDegradation={})".format(degradations[t]))
                    if (t==tag):
                        # ablation tag, so use bolder symbol
                        plt.plot(degradations[t], 'bs', label=t)
                    else:
                        plt.plot(degradations[t], label=t)

                plt.title(tag, fontsize=18)
                plt.legend()
                plt.savefig("{}_chart.png".format(tag))
                plt.clf()  # clear the plot -was plot.show()

    elif status == 'decode':
        print("MODEL: decode")
        data.load(data.dset_dir)
        data.read_config(args.config)
        print(data.raw_dir)
        # exit(0)
        data.show_data_summary()
        data.generate_instance('raw')
        print("nbest: %s"%(data.nbest))
        decode_results, pred_scores = load_model_decode(data, 'raw')
        if data.nbest and not data.sentence_classification:
            data.write_nbest_decoded_results(decode_results, pred_scores, 'raw')
        else:
            data.write_decoded_results(decode_results, 'raw')
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")

