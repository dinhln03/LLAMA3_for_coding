#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Script to Test Deep Learning Model.

Contains a pipeline to test a deep learning model.

Revision History:
        2021-11-20 (ANI717 - Animesh Bala Ani): Baseline Software.

Example:
        $ python3 test.py

"""


#___Import Modules:
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from model import NvidiaNet
from dataset import ANI717Dataset


#___Main Method:
def main():
    
    # Load Data
    dataset = ANI717Dataset(config.TEST_CSV, config.IMG_SOURCE, transforms=config.TEST_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Initialize Model with Weights
    model = NvidiaNet(in_channels=config.IMG_SHAPE[0]).to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_FILE, map_location=config.DEVICE)["state_dict"])
    model.eval()
    
    # Initialize total correct number and counter
    num_correct = 0.0
    count = 0
    
    # Loop through dataset
    with torch.no_grad():
        
        loop = tqdm(loader, position=0, leave=True)
        for batch_idx, (inputs, z, x) in enumerate(loop):
            
            # Enable GPU support is available
            inputs = inputs.to(config.DEVICE)
            if config.TRAIN_TYPE == 'z':
                targets = z.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            else:
                targets = x.unsqueeze(1).to(torch.float32).to(config.DEVICE)
            
            # Calculate prediction
            predictions = model(inputs)
            
            # Update total correct number and counter
            num_correct += sum(abs(torch.round(targets/config.ERROR_TOLERENCE) - torch.round(predictions/config.ERROR_TOLERENCE)) <= 1).item()
            count += predictions.shape[0]
            
            # Calculate accuracy
            loop.set_postfix(accuracy=100*num_correct/count)


#___Driver Program:
if __name__ == "__main__":
    main()


#                                                                              
# end of file
"""ANI717"""