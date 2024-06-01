import pickle
import os
from tqdm import tqdm

with open('../data/bawe_splits.p', 'rb') as f:
    splits = pickle.load(f)

if not os.path.isdir('../data/preprocess/bawe-group'):
    os.mkdir('../data/preprocess/bawe-group')

for filename in tqdm(splits['train']):
    id = filename[:4]
    with open(f'../data/bawe/CORPUS_TXT/{filename}', 'r') as f:
        if not os.path.isdir(f'../data/preprocess/bawe-group/{id}'):
            os.mkdir(f'../data/preprocess/bawe-group/{id}')

        text = f.read()

        with open(f'../data/preprocess/bawe-group/{id}/{filename}', 'w') as wf:
            wf.write(text)
