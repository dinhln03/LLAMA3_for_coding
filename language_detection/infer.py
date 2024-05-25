from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
import torch
from torch.utils.data import DataLoader,SequentialSampler
import torch.nn.functional as F
import os
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.multiprocessing as mp
from multiprocessing import Queue
import json
from tqdm import tqdm

#------------------Helper Function-----------------#
def load_and_tokenize_dataset(tokenizer):
    raw_data = load_dataset("sahil2801/CodeAlpaca-20k")
    def convert_to_ids(examples):
          return tokenizer(examples["output"], truncation = True, padding = "max_length", max_length = 512, pad_to_multiple_of=8, add_special_tokens=True)
    raw_data = raw_data.map(convert_to_ids, batched=True)
    return raw_data
#---------------------------------------------------#
    
class ModelLoader:
    def __init__(self, rank: int, model_name_or_path: str = "philomath-1209/programming-language-identification"):
        self.rank = rank
        self.model_name_or_path = model_name_or_path

    def load_model_and_tokenizer(self):
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name_or_path, device_map=f"cuda:{self.rank}")
            tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True, padding_side='right')
            return model, tokenizer
        except Exception as e:
            print(f"An error occurred while loading the model or tokenizer: {e}")
            return None, None
    
class LanguageDetection:
    def __init__(self, tokenized_dataset: Dataset):
        self.tokenized_dataset = tokenized_dataset
        self.tokenized_dataset.set_format(
                type="torch", columns=["input_ids", "attention_mask"],
        )
        self.id2label = {0: 'Scala', 1: 'JavaScript', 2: 'COBOL', 3: 'ARM Assembly', 4: 'R', 5: 'Lua', 6: 'C++', 7: 'Visual Basic .NET', 
             8: 'Go', 9: 'Erlang', 10: 'C#', 11: 'Rust', 12: 'Ruby', 13: 'Swift', 14: 'Mathematica/Wolfram Language', 
             15: 'PHP', 16: 'Fortran', 17: 'AppleScript', 18: 'Pascal', 19: 'Java', 20: 'PowerShell', 21: 'Python', 
             22: 'C', 23: 'Perl', 24: 'Kotlin', 25: 'jq'}
        
    def compute_labels(self, model: PreTrainedModel, rank):
        labels = []
        test_dataloader = DataLoader(self.tokenized_dataset, batch_size=256, drop_last = False, sampler= SequentialSampler(self.tokenized_dataset),pin_memory=True
                                    ,num_workers=4)
        with torch.no_grad():
            for batch in tqdm(test_dataloader, position = rank, desc=f"Process {rank}"):
                input_ids, attention_masks = (batch["input_ids"].to(f"cuda:{rank}"), batch["attention_mask"].to(f"cuda:{rank}")) 
                batch_logits = model(input_ids, attention_mask=attention_masks).logits
                batch_logit_probs = F.softmax(batch_logits, dim = 1).max(dim = 1)
                batch_ids = batch_logit_probs[1]    #ids : 1,2,3
                batch_probs = batch_logit_probs[0]  #probs: 0.5, 0.6
                batch_ids = torch.where(batch_probs <= 0.6, -100, batch_ids)
                batch_labels = list(map(self.id2label.get, batch_ids.tolist()))
                labels.extend(batch_labels)
        return labels  


def run(rank,queue):
    model, tokenizer = ModelLoader(rank).load_model_and_tokenizer()
    tokenized_dataset = load_and_tokenize_dataset(tokenizer)
    split = len(tokenized_dataset["train"]) //2
    if rank ==0:
        tokenized_dataset = tokenized_dataset["train"].select(range(split))
        label_from_rank1 = []
    if rank ==1:
        tokenized_dataset = tokenized_dataset["train"].select(range(split,len(tokenized_dataset["train"])))
    labels = LanguageDetection(tokenized_dataset).compute_labels(model, rank)
    queue.put((rank, labels))
    
    
def init_process(rank, size, fn,queue, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '5100'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank,queue)

if __name__ == "__main__":
    size = 2
    processes = []
    mp.set_start_method("spawn")
    queue = Queue()
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run,queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        
    lang = []
    for _ in range(size):
        rank_,labels_ = queue.get()
        lang += labels_
        del labels_, rank_
    print("Len language:", len(lang))
    json_data = {"lang":lang}
    with open("processed_data/language.json","w") as f:
        json.dump(json_data,f)
    print("Done")   # Happened in main process
