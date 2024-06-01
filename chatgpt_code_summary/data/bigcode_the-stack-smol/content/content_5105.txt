import torch;
import numpy as np;
from torch import nn;
from torch import optim;
import torch.functional as func;
from torchvision import datasets, transforms, models;
import time;
from os import path;
import argparse;
import utils
import json;

def main(test_image_path, checkpoint_path, top_k, category_names, gpu):
    print("test_image_path: " , test_image_path);
    print("checkpoint_path: " , checkpoint_path);
    print("top_k: " , top_k);
    print("category_names: " , category_names);
    print("Use_GPU" , gpu);

    if gpu == True:
        device='cuda';
    else:
        device='cpu';
    
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = None;
    
    model, class_to_idx = utils.load_model_checkpoint(checkpoint_path);
    
    idx_to_class = {class_to_idx[k] : k for k in class_to_idx};
    
    probs, outputs, names = utils.predict(test_image_path, model, idx_to_class, cat_to_name, top_k, device);

    print(f'The Top {top_k} predictions are:');
    for i in range(top_k):
        print(f'\tPrediction #{i} : {names[i]} with Probability : {probs[i]:.3f}');
        
if __name__ == '__main__':
    print('qaz');
    
    parser = argparse.ArgumentParser(description='Image Classification Project')
    parser.add_argument('test_image_path', action="store", help="Location of Test file for predicting classes of");
    parser.add_argument('checkpoint_path', action="store", help="Location of Model Checkpoint file (must have file format .pth)");
    parser.add_argument('--top_k', action="store", dest="top_k", help="Number of Top Likely classes predicted.", default=3, type=int)
    parser.add_argument('--category_names', action="store", dest="category_names", help="path to a file with class categories to real names", default="cat_to_name.json");
    parser.add_argument('--gpu', action="store_true", dest="gpu", default=False, help="is provided CUDA gpu will be used, else CPU")
    
    parsed_args = parser.parse_args();

    main(parsed_args.test_image_path, parsed_args.checkpoint_path, parsed_args.top_k, parsed_args.category_names, parsed_args.gpu);