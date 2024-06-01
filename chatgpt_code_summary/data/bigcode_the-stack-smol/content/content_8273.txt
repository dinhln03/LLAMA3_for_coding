import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
from keras import backend as K
from keras.utils import Sequence
import keras.utils
import tensorflow as tf
import multi_utils
import mat_utils

class NbestFixedDataGenerator(Sequence):

    def __init__(self, file, key_file, batch_size=64, feat_dim=40, n_labels=1024,
                 procs=10, extras1=10, extras2=10, num_extras1=1, nbest=100, mode='train', shuffle=False,
                 mod=1):

        self.file=file
        self.batch_size=batch_size
        self.feat_dim=feat_dim
        self.n_labels=n_labels
        self.procs=procs
        self.extras1=extras1
        self.extras2=extras2
        self.num_extras1=num_extras1
        self.nbest=nbest
        self.shuffle=shuffle
        self.keys=[]
        self.sorted_keys=[]
        self.mode=mode
        self.mod=1

        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        if key_file is not None:
            with open(key_file, 'r') as f:
                for line in f:
                    self.sorted_keys.append(line.strip())
        for key in self.h5fd.keys():
            self.keys.append(key)

        self.n_samples = len(self.h5fd.keys())
        for key in self.h5fd.keys():
            self.keys.append(key)
        if len(self.sorted_keys) > 0:
            self.keys = self.sorted_keys

    def __len__(self):
        return int(np.ceil(self.n_samples)/self.batch_size)

    def __getitem__(self, index, return_keys=False):
        list_keys_temp = [self.keys[k] for k in range(index*self.batch_size,
                                                      min( (index+1)*self.batch_size,
                                                      len(self.keys) ) )]

        # [input_sequences, label_sequences, inputs_lengths, labels_length]
        if self.mode == 'train':
            x, mask, y = self.__data_generation(list_keys_temp)
            if return_keys == True:
                return x, mask, y, list_keys_temp
            else:
                return x, mask, y
        else:
            x, mask = self.__data_generation(list_keys_temp)
            if return_keys == True:
                return x, mask, list_keys_temp
            else:
                return x, mask

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.keys)

    def __data_generation(self, list_keys_temp):

        max_num_blocks=0
        max_num_frames=0

        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            mat = mat_utils.pad_mat(mat, self.mod)
            [ex_blocks,ex_frames] = multi_utils.expected_num_blocks(mat,
                                                                    self.procs,
                                                                    self.extras1,
                                                                    self.extras2,
                                                                    self.num_extras1)
            if ex_blocks > max_num_blocks:
                max_num_blocks = ex_blocks
            if ex_frames > max_num_frames:
                max_num_frames = ex_frames

        input_mat=np.zeros((len(list_keys_temp), max_num_blocks,
                            self.procs+max(self.extras1, self.extras2), self.feat_dim))
        input_mask=np.zeros((len(list_keys_temp), max_num_blocks,
                             self.procs+max(self.extras1, self.extras2), 1))
        if self.mode == 'train':
            numer_labels=np.zeros((len(list_keys_temp), max_num_blocks,
                                   self.procs+max(self.extras1, self.extras2), self.n_labels+1))
            numer_lmscores = np.zros((len(list_keys_temp), 1))

            denom_labels=np.zeros((len(list_keys_temp), self.nbest, max_num_blocks,
                                   self.procs+max(self.extras1, self.extras2), self.n_labels+1))
            denom_lmlscores = np.zeros((len(list_keys_temp), self.nbest, 1))

        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            [ex_blocks, ex_frames] = multi_utils.expected_num_blocks(mat,
                                                                     self.procs,
                                                                     self.extras1,
                                                                     self.extras2,
                                                                     self.num_extras1)
            blocked_mat, mask , _ = multi_utils.split_utt(mat, self.procs, self.extras1,
                                                                   self.extras2,
                                                                   self.num_extras1,
                                                                   ex_blocks,
                                                                   self.feat_dim, max_num_blocks)
            input_mat[i,:,:,:] = np.expand_dims(blocked_mat, axis=0)
            input_mask[i,:,:,:] = np.expand_dims(mask, axis=0)

            if self.mode == 'train':
                # label is a list of string starting from 0
                numer = self.h5fd[key+'/1best'][()]
                numer_labels = multi_utils.str2dict(numer)
                numer_lmscores[i,0] = self.h5fd[key+'/1best_scores'][()]

                denom = self.h5fd[key+'/nbest'][()]
                denom_labels = multi_utils.str2nbest(denom)
                denom_lmscores[i, :, 0] = self.h5fd[key+'/nbest_scores'][()]

                # w/o padding for convenience
                # splitting labels
                # (blocks, frames, feats)
                number_blocked_labels = multi_utils.split_post_label(numer_labels, self.procs, self.extras1,
                                                                     self.extras2, self.num_extras1, ex_blocks,
                                                                     self.n_labels+1, max_num_blocks)
                # expand dimensions along with batch dim.
                numer_labels[i,:,:,:] = np.expand_dims(numer_blocked_labels, axis=0)

                # (nbest, blocks, time, feats)
                denom_blocked_labels = muti_utils.split_nbest_label(denom_labels, self.procs, self.extra1,
                                                                    self.extra2, self.num_extras1, ex_blocks,
                                                                    self.n_labels+1, max_num_blocks)
                denom_labels[i,:,:,:,:] = np.expand_dims(denom_blocked_labels, axis=0)

        # transpose batch and block axes for outer loop in training
        input_mat = input_mat.transpose((1,0,2,3))
        input_mask = input_mask.transpose((1,0,2,3))
        if self.mode == 'train':
            # transpose batch dim. <-> block dim.
            number_labels = numer_labels.transpose((1,0,2,3)) # (batch,, blocks, time, feats) -> (blocks, batch, time, feats)
            denom_labels = denom_labels.transpose((2,1,0,3,4)) # (batch, nbest, blocks, time, feats)->(nbest, blocks, batch, time, feats)

        if self.mode == 'train':
            return input_mat, input_mask, [numer_labels, numer_lmscores, denom_labels, denom_lmscores]
        else:
            return input_mat, input_mask
