#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import sys, os
import pandas as pd 
import torch
from torchsummary import summary
from torchtext import data
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, TensorDataset,DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import pickle
import shutil
from sklearn.model_selection import train_test_split


def tokenize(tokenizer,text_array,max_seq_len=64,pad_to_max_length=True,add_special_tokens=True):
    ''' Returns tokenized IDs and attention mask
    The transformers encode_plus method returns the following:
    {
    input_ids: list[int],
    token_type_ids: list[int] if return_token_type_ids is True (default)
    attention_mask: list[int] if return_attention_mask is True (default)
    overflowing_tokens: list[int] if a ``max_length`` is specified and return_overflowing_tokens is True
    num_truncated_tokens: int if a ``max_length`` is specified and return_overflowing_tokens is True
    special_tokens_mask: list[int] if ``add_special_tokens`` if set to ``True`` and return_special_tokens_mask is True
    }'''
    all_tokens=[]
    all_attention_mask=[]
    for i,text in enumerate(tqdm(text_array)):
        encoded = tokenizer.encode_plus(
                       text,
                       add_special_tokens=add_special_tokens,
                       max_length=max_seq_len,
                       pad_to_max_length=pad_to_max_length)
        tokens = torch.tensor(encoded['input_ids'])
        attention_mask = torch.tensor(encoded['attention_mask'])
        all_tokens.append(tokens)
        all_attention_mask.append(attention_mask)
    return all_tokens,all_attention_mask

class CreateDataset(Dataset):
    def __init__(self,data,atten_mask,labels,num_excl):
        self._dataset = [[data[i],atten_mask[i],labels.values[i],num_excl.values[i]] for i in range(0,len(data))]
    
    def __len__(self):
        return len(self._dataset)

    def __getitem__(self,idx):
        return self._dataset[idx]

def createTestTrainSplit(all_train_df,test_size=0.2,seed=1234):
    # Create train, validation dataset splits
    train_df, valid_df = train_test_split(all_train_df, test_size=0.2,random_state=seed)
    train_data   = train_df.text.fillna("DUMMY_VALUE")
    train_labels = train_df.label
    train_num_excl = train_df.num_exclamation_marks
    valid_data  = valid_df.text.fillna("DUMMY_VALUE")
    valid_labels = valid_df.label
    valid_num_excl = train_df.num_exclamation_marks
    return train_data,train_labels,train_num_excl,valid_data,valid_labels,valid_num_excl

def saveTokensToFiles(TOKEN_DATA_PATH,
                    train_data_tokenized,train_attention_mask,
                    valid_data_tokenized,valid_attention_mask,
                    test_data_tokenized,test_attention_mask):
     # save to files for later use
     with open(TOKEN_DATA_PATH+'/train_data_tokenized.txt', 'wb') as fp:
         pickle.dump(train_data_tokenized, fp)
     with open(TOKEN_DATA_PATH+'/train_attention_mask.txt', 'wb') as fp:
         pickle.dump(train_attention_mask, fp)
     with open(TOKEN_DATA_PATH+'/valid_data_tokenized.txt', 'wb') as fp:
         pickle.dump(valid_data_tokenized, fp)
     with open(TOKEN_DATA_PATH+'/valid_attention_mask.txt', 'wb') as fp:
         pickle.dump(valid_attention_mask, fp)
     with open(TOKEN_DATA_PATH+'/test_data_tokenized.txt', 'wb') as fp:
         pickle.dump(test_data_tokenized, fp)
     with open(TOKEN_DATA_PATH+'/test_attention_mask.txt', 'wb') as fp:
         pickle.dump(test_attention_mask, fp)

def loadTokensFromFiles(TOKEN_DATA_PATH,
                        train_data_tokenized,train_attention_mask,
                        valid_data_tokenized,valid_attention_mask,
                        test_data_tokenized,test_attention_mask):
     # read back tokenized data
     with open(TOKEN_DATA_PATH+'train_data_tokenized.txt', 'rb') as fp:
         train_data_tokenized=pickle.load(fp)
     with open(TOKEN_DATA_PATH+'train_attention_mask.txt', 'rb') as fp:
         train_attention_mask=pickle.load(fp)
     with open(TOKEN_DATA_PATH+'valid_data_tokenized.txt', 'rb') as fp:
         valid_data_tokenized=pickle.load(fp)
     with open(TOKEN_DATA_PATH+'valid_attention_mask.txt', 'rb') as fp:
         valid_attention_mask=pickle.load(fp)
     with open(TOKEN_DATA_PATH+'test_data_tokenized.txt', 'rb') as fp:
         test_data_tokenized=pickle.load(fp)
     with open(TOKEN_DATA_PATH+'test_attention_mask.txt', 'rb') as fp:
         test_attention_mask=pickle.load(fp)

def generateDataLoader(dataset,batch_size,shuffle=False,num_workers=16,pin_memory=False,drop_last=True):
    # print("Expected number of batches:", int(len(train_data_tokenized)/params['batch_size']))
    sampler = RandomSampler(dataset)
    dataLoader = torch.utils.data.DataLoader(dataset=dataset,
                                             sampler=sampler,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers)
    return dataLoader
