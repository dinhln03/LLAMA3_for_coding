# MIT License
#
# Copyright (c) 2017 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Created: 2018-09-04
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0', input_size=1):

        super(TextGenerationModel, self).__init__()
        
        self.emb_size = 64
        
        self.device = device
        # self.emb = nn.Embedding(batch_size * seq_length, 64)
        # self.lstm = nn.LSTM(64, lstm_num_hidden, num_layers=lstm_num_layers, dropout=0)
        self.lstm = nn.LSTM(input_size, lstm_num_hidden, num_layers=lstm_num_layers, dropout=0)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        self.h = None

    def forward(self, x):
        
        # Reset hidden layer for Training
        if self.training:
            self.h = None
            
        # x = self.emb(x.squeeze(-1).type(torch.LongTensor).to(self.device))
        
        out, h = self.lstm(x.transpose(0, 1), self.h)
        out = self.linear(out)

        # Handle hidden layer for Inference
        if not self.training:
            self.h = h
        
        return out
    
    def reset_hidden(self):
        self.h = None
