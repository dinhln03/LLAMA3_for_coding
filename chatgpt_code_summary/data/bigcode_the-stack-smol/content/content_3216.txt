from sklearn.cluster import MiniBatchKMeans
import numpy as np

import torch
from models import TransformerModel, Seq2SeqTransformer, generate_square_subsequent_mask
from models import LM_NAME, MLM_NAME, MT_NAME, NLAYERS, NUM2WORD
import os
from data_preprocessing import DATA_DIR_DEV, SAVE_DATA_MT_TRAIN
from data_preprocessing import SAVE_VOCAB_SRC, SAVE_VOCAB_TRG, PAD_WORD
import pickle
from torchtext.legacy.data import Dataset, BucketIterator
import pandas as pd
from analytics_helper import MostFreqToken, GetInter, GetMI, GetInterValues
from analytics_helper import MIN_SAMPLE_SIZE_DEV, MIN_SAMPLE_SIZE_FULL
from analytics_helper import N_FREQUENT_DEV, N_FREQUENT_FULL
from analytics_helper import N_CLUSTER_DEV, N_CLUSTER_FULL
from data_preprocessing import SAVE_MODEL_PATH, DEVELOPMENT_MODE
from MT_helpers import patch_trg, create_mask


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if DEVELOPMENT_MODE:
    min_sample_size=MIN_SAMPLE_SIZE_DEV
    N_frequent=N_FREQUENT_DEV
    N_cluster=N_CLUSTER_DEV
    data_dir=DATA_DIR_DEV

else:
    min_sample_size=MIN_SAMPLE_SIZE_FULL
    N_frequent=N_FREQUENT_FULL
    N_cluster=N_CLUSTER_FULL
    data_dir=DATA_DIR_FULL


MI_results_INP={LM_NAME.split('.')[0]:[],
         f"{MLM_NAME.split('.')[0]}_SAME":[],
         f"{MLM_NAME.split('.')[0]}_DIFF":[],
         MT_NAME.split('.')[0]:[]}

MI_results_OUT={LM_NAME.split('.')[0]:[],
         MLM_NAME.split('.')[0]:[]}

MODELS_INP=[LM_NAME, MLM_NAME, MT_NAME]

vocab_pkl_src = os.path.join(data_dir, SAVE_VOCAB_SRC)
vocab_pkl_trg = os.path.join(data_dir, SAVE_VOCAB_TRG)
train_pkl = os.path.join(data_dir, SAVE_DATA_MT_TRAIN)
field_src = pickle.load(open(vocab_pkl_src, 'rb'))
field_trg = pickle.load(open(vocab_pkl_trg, 'rb'))
src_pad_idx = field_src.vocab.stoi[PAD_WORD]
trg_pad_idx = field_trg.vocab.stoi[PAD_WORD]
train_examples = pickle.load(open(train_pkl, 'rb'))
fields = {'src':field_src , 'trg':field_trg}
train = Dataset(examples=train_examples, fields=fields)
train_iter = BucketIterator(train, batch_size=1, device=device, train=True, shuffle=False)
frequent_vocab = MostFreqToken(field_src, N_frequent, min_sample_size)

# token_reps_list saves NLAYERS dicts, for ith dict, the key is the token ID,
# the value is the representation of the ID in the ith layer.
token_reps_model_INP={}
token_reps_model_OUT={}
for this_model_name in MODELS_INP:
    token_reps_list=[]
    for _ in range(NLAYERS):
        this_token_reps={}
        for this_token_id in frequent_vocab:
            this_token_reps[this_token_id]=[]
        token_reps_list.append(this_token_reps)
    if this_model_name.startswith("MLM"):
        token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_SAME"]=token_reps_list
        token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_DIFF"]=token_reps_list
        token_reps_model_OUT[this_model_name.split('.')[0]]=token_reps_list
    elif this_model_name.startswith("LM"):
        token_reps_model_INP[this_model_name.split('.')[0]]=token_reps_list
        token_reps_model_OUT[this_model_name.split('.')[0]]=token_reps_list
    elif this_model_name.startswith("MT"):
        token_reps_model_INP[this_model_name.split('.')[0]]=token_reps_list

sample_size_dict_INP={}
sample_size_dict_OUT={}
for this_model_name in MODELS_INP:
    if this_model_name.startswith("MLM"):
        this_sample_size_dict_INP_SAME={}
        this_sample_size_dict_INP_DIFF={}
        this_sample_size_dict_OUT={}
        for this_token_id in frequent_vocab:
            this_sample_size_dict_INP_SAME[this_token_id]=0
            this_sample_size_dict_INP_DIFF[this_token_id]=0
            this_sample_size_dict_OUT[this_token_id]=0
        sample_size_dict_INP[f"{this_model_name.split('.')[0]}_SAME"]=this_sample_size_dict_INP_SAME
        sample_size_dict_INP[f"{this_model_name.split('.')[0]}_DIFF"]=this_sample_size_dict_INP_DIFF
        sample_size_dict_OUT[this_model_name.split('.')[0]]=this_sample_size_dict_OUT
    elif this_model_name.startswith("LM"):
        this_sample_size_dict_INP={}
        this_sample_size_dict_OUT={}
        for this_token_id in frequent_vocab:
            this_sample_size_dict_INP[this_token_id]=0
            this_sample_size_dict_OUT[this_token_id]=0
        sample_size_dict_INP[this_model_name.split('.')[0]]=this_sample_size_dict_INP
        sample_size_dict_OUT[this_model_name.split('.')[0]]=this_sample_size_dict_OUT
    elif this_model_name.startswith("MT"):
        this_sample_size_dict_INP={}
        for this_token_id in frequent_vocab:
            this_sample_size_dict_INP[this_token_id]=0
        sample_size_dict_INP[this_model_name.split('.')[0]]=this_sample_size_dict_INP



for batch in train_iter:
    src_seq_MT = batch.src.to(device)
    target_sample_INP_MT=GetInter(src_seq_MT.detach().numpy(), frequent_vocab)

    src_seq_MLM_SAME = batch.src.to(device)
    target_sample_INP_MLM_SAME=GetInter(src_seq_MLM_SAME.detach().numpy(), frequent_vocab)

    src_seq=batch.src.to(device)
    src_seq_MLM_DIFF = src_seq.clone()
    src_mask = generate_square_subsequent_mask(src_seq.size(0))
    rand_value = torch.rand(src_seq.shape)
    rand_mask = (rand_value < 0.15) * (input != src_pad_idx)
    mask_idx=(rand_mask.flatten() == True).nonzero().view(-1)
    src_seq_MLM_DIFF = src_seq_MLM_DIFF.flatten()
    src_seq_MLM_DIFF[mask_idx] = 103
    src_seq_MLM_DIFF = src_seq_MLM_DIFF.view(src_seq.size())
    target_sample_INP_MLM_DIFF=GetInter(src_seq_MLM_DIFF.detach().numpy(), frequent_vocab)

    src_seq_LM = batch.src[:-1]
    target_sample_INP_LM=GetInter(src_seq_LM.detach().numpy(), frequent_vocab)

    trg = batch.trg
    trg_seq_MT, gold = map(lambda x: x.to(device), patch_trg(trg, trg_pad_idx))
    trg_seq_MT = trg_seq_MT.to(device)

    trg_seq_LM = src_seq[1:].to(device)
    target_sample_OUT_LM=GetInter(trg_seq_LM.detach().numpy(), frequent_vocab)

    trg_seq_MLM = src_seq
    target_sample_OUT_MLM=GetInter(trg_seq_MLM.detach().numpy(), frequent_vocab)

    for this_model_name in MODELS_INP:
        this_model = torch.load(os.path.join(SAVE_MODEL_PATH,this_model_name))
        this_model.eval()
        if this_model_name.startswith("MT") and len(target_sample_INP_MT)>0:
            src_mask, trg_mask, src_padding_mask, trg_padding_mask = create_mask(src_seq_MT, trg_seq_MT, src_pad_idx, trg_pad_idx)
            _ = this_model(src=src_seq_MT,
                           src_mask=src_mask,
                           trg=trg_seq_MT,
                           tgt_mask=trg_mask,
                           src_padding_mask=src_padding_mask,
                           tgt_padding_mask=trg_padding_mask,
                           memory_key_padding_mask=src_padding_mask)
            token_reps_list=token_reps_model_INP[MT_NAME.split('.')[0]]
            this_sample_size_dict=sample_size_dict_INP[this_model_name.split('.')[0]]
            GetInterValues(this_model, target_sample_INP_MT, NUM2WORD, token_reps_list, this_sample_size_dict, min_sample_size, NLAYERS)
        elif this_model_name.startswith("MLM"):
            if len(target_sample_INP_MLM_SAME)>0:
                src_mask = generate_square_subsequent_mask(src_seq_MLM_SAME.size(0))
                src_padding_mask = (src_seq_MLM_SAME == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_MLM_SAME, src_mask.to(device),src_padding_mask.to(device))
                token_reps_list=token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_SAME"]
                this_sample_size_dict=sample_size_dict_INP[f"{this_model_name.split('.')[0]}_SAME"]
                GetInterValues(this_model, target_sample_INP_MLM_SAME, NUM2WORD, token_reps_list, this_sample_size_dict, min_sample_size, NLAYERS)

            if len(target_sample_INP_MLM_DIFF)>0 and len(target_sample_OUT_MLM)>0:
                src_mask = generate_square_subsequent_mask(src_seq_MLM_DIFF.size(0))
                src_padding_mask = (src_seq_MLM_DIFF == src_pad_idx).transpose(0, 1)
                _ = this_model(src_seq_MLM_DIFF.to(device), src_mask.to(device),src_padding_mask.to(device))
                token_reps_list_INP=token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_DIFF"]
                this_sample_size_dict_INP=sample_size_dict_INP[f"{this_model_name.split('.')[0]}_DIFF"]

                token_reps_list_OUT=token_reps_model_OUT[MLM_NAME.split('.')[0]]
                this_sample_size_dict_OUT=sample_size_dict_OUT[this_model_name.split('.')[0]]

                GetInterValues(this_model, target_sample_INP_MLM_DIFF, NUM2WORD, token_reps_list_INP, this_sample_size_dict_INP, min_sample_size, NLAYERS)
                GetInterValues(this_model, target_sample_OUT_MLM, NUM2WORD, token_reps_list_OUT, this_sample_size_dict_OUT, min_sample_size, NLAYERS)
        elif this_model_name.startswith("LM") and len(target_sample_INP_LM)>0 and len(target_sample_OUT_LM)>0:
            src_mask = generate_square_subsequent_mask(src_seq_LM.size(0))
            src_padding_mask = (src_seq_LM == src_pad_idx).transpose(0, 1)
            _ = this_model(src_seq_LM, src_mask.to(device),src_padding_mask.to(device))
            token_reps_list_INP=token_reps_model_INP[this_model_name.split('.')[0]]
            token_reps_list_OUT=token_reps_model_OUT[this_model_name.split('.')[0]]

            this_sample_size_dict_INP=sample_size_dict_INP[this_model_name.split('.')[0]]
            this_sample_size_dict_OUT=sample_size_dict_OUT[this_model_name.split('.')[0]]

            GetInterValues(this_model, target_sample_INP_LM, NUM2WORD, token_reps_list_INP, this_sample_size_dict_INP, min_sample_size, NLAYERS)
            GetInterValues(this_model, target_sample_OUT_LM, NUM2WORD, token_reps_list_OUT, this_sample_size_dict_OUT, min_sample_size, NLAYERS)


    # we only need to keep the minimum sample size that has been collected
    this_min_sample_size_inp=float('inf')
    this_min_sample_size_out=float('inf')

    for model_name, this_sample_size_dict in sample_size_dict_INP.items():
        for token_id, size in this_sample_size_dict.items():
            if size<this_min_sample_size_inp:
                this_min_sample_size_inp=size

    for model_name, this_sample_size_dict in sample_size_dict_OUT.items():
        for token_id, size in this_sample_size_dict.items():
            if size<this_min_sample_size_out:
                this_min_sample_size_out=size

    is_enough=True
    if this_min_sample_size_inp>=min_sample_size and this_min_sample_size_out>=min_sample_size:
        for model_name, reps_dict in token_reps_model_INP.items():
            if is_enough is False:
                break
            for this_layer in reps_dict:
                if is_enough is False:
                    break
                for token_id, rep_list in this_layer.items():
                    if len(rep_list)<min_sample_size:
                        is_enough=False
                        break

        for model_name, reps_list in token_reps_model_OUT.items():
            if is_enough is False:
                break
            for this_layer in reps_dict:
                if is_enough is False:
                    break
                for token_id, rep_list in this_layer.items():
                    if len(rep_list)<min_sample_size:
                        is_enough=False
                        break
    else:
        is_enough=False
    if is_enough:
        break

if is_enough is False:
    assert 1==0, "We have not collected enough data!"

for this_model_name in MODELS_INP:
    if this_model_name.startswith("MLM"):
        token_reps_list=token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_SAME"]
        result_list=MI_results_INP[f"{MLM_NAME.split('.')[0]}_SAME"]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

        token_reps_list=token_reps_model_INP[f"{MLM_NAME.split('.')[0]}_DIFF"]
        result_list=MI_results_INP[f"{MLM_NAME.split('.')[0]}_DIFF"]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

        token_reps_list=token_reps_model_OUT[MLM_NAME.split('.')[0]]
        result_list=MI_results_OUT[MLM_NAME.split('.')[0]]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

    elif this_model_name.startswith("MT"):
        token_reps_list=token_reps_model_INP[this_model_name.split('.')[0]]
        result_list=MI_results_INP[this_model_name.split('.')[0]]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

    elif this_model_name.startswith("LM"):
        token_reps_list=token_reps_model_INP[this_model_name.split('.')[0]]
        result_list=MI_results_INP[this_model_name.split('.')[0]]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)

        token_reps_list=token_reps_model_OUT[MLM_NAME.split('.')[0]]
        result_list=MI_results_OUT[this_model_name.split('.')[0]]
        GetMI(token_reps_list, N_frequent, N_cluster, NLAYERS, result_list)



print("result",MI_results_INP)
print("result",MI_results_OUT)
