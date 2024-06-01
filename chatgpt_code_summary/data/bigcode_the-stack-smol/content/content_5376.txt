import numpy as np
from torch import nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
import pandas as pd
import data_loader as dl
import time
import copy
import utility
import yaml
import trainer
from PIL import Image
from os import path
Image.MAX_IMAGE_PIXELS = None
from scipy.io import savemat
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os.path
from os import path

BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001
ANNEAL_STRAT = "cos"
FEATURE_EXTRACT = False
APPLY_ZCA_TRANS = True
DATA_DIR = 'data/train_images'
NETS = ['resnext'] # train on resnext
IMAGE_SIZES = [64, 128, 224] # train for 4 resolutions

def main():
    # Load the meta data file
    df = pd.read_csv('./data/train.csv')
    df, label_encoder = utility.encode_labels(df)
    num_classes = len(df['label'].value_counts())
    np.save('./data/label_encoder_classes.npy', label_encoder.classes_)
        
    # Generate the ZCA matrix if enabled
    for image_size in IMAGE_SIZES: # train for every res
        if APPLY_ZCA_TRANS:
            print("Making ZCA matrix ...")
            data_loader = dl.get_full_data_loader(df, data_dir=DATA_DIR,
                                                batch_size=BATCH_SIZE,
                                                image_size=image_size)
            train_dataset_arr = next(iter(data_loader))[0].numpy()
            zca = utility.ZCA()
            zca.fit(train_dataset_arr)
            zca_dic = {"zca_matrix": zca.ZCA_mat, "zca_mean": zca.mean}
            savemat("./data/zca_data.mat", zca_dic)
            print("Completed making ZCA matrix")

        # Define normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        # Define specific transforms
        train_transform = transforms.Compose([
                utility.AddPadding(),
                transforms.Resize((image_size,image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=(-90, 90)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(.4,.4,.4),
                transforms.ToTensor(),
                normalize
            ])
        valid_transform = transforms.Compose([
                utility.AddPadding(),
                transforms.Resize((image_size,image_size)),
                transforms.ToTensor(),
                normalize
        ])
    
        # Create a train and valid dataset
        train_dataset = dl.HotelImagesDataset(df, root_dir=DATA_DIR,
                                            transform=train_transform)
        valid_dataset = dl.HotelImagesDataset(df, root_dir=DATA_DIR,
                                            transform=valid_transform)
            
        # Get a train and valid data loader
        train_loader, valid_loader = dl.get_train_valid_loader(train_dataset,
                                                            valid_dataset,
                                                            batch_size=BATCH_SIZE,
                                                            random_seed=0)
        for net_type in NETS: # train for every net
            model = utility.initialize_net(num_classes, net_type,
                                           feature_extract=FEATURE_EXTRACT)
            
            # If old model exists, take state from it
            if path.exists(f"./models/model_{net_type}.pt"):
                print("Resuming training on trained model ...")
                model = utility.load_latest_model(model, f'./models/model_{net_type}.pt')
                
            # Gather the parameters to be optimized/updated in this run.
            params_to_update = utility.get_model_params_to_train(model, FEATURE_EXTRACT)
        
            # Send model to GPU
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")            
            model = model.to(device)

            # Make criterion
            criterion = nn.CrossEntropyLoss()
            
            # Make optimizer + scheduler
            optimizer = torch.optim.SGD(params_to_update, lr=0.01, momentum=0.9)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                   mode='min',
                                                                   factor=0.01,
                                                                   patience=3)

            trained_model = trainer.train_model(device=device,
                                                model=model,
                                                optimizer=optimizer,
                                                criterion=criterion,
                                                train_loader=train_loader,
                                                valid_loader=valid_loader,
                                                scheduler=scheduler,
                                                net_type=net_type,
                                                epochs=EPOCHS,
                                                apply_zca_trans=APPLY_ZCA_TRANS)
        
            utility.save_current_model(trained_model,
                                    f"./models/model_{net_type}.pt")
                    
if __name__ == "__main__":
    main()