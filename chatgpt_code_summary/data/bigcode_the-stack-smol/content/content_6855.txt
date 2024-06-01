#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2020 Hnaynag University (Jae-Hong Lee)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import codecs
import json
import logging
import re
import random

from pathlib import Path

from tqdm import tqdm
from nltk import tokenize

from espnet.utils.cli_utils import get_commandline_args

def error_checker(keys, file_path, log_path):
    buffer_key = None
    past_key = None
    total_key_count = len(keys)
    skip_key_count = 0 
    with open(file_path, encoding="utf-8") as f:
        for line in tqdm(f.readlines()):
            sps = line.rstrip().split(maxsplit=1)
            if len(sps) == 2:
                key, value = sps
                if key in keys:
                    past_key = key
                else:
                    if buffer_key != past_key:
                        keys.remove(past_key)
                        skip_key_count += 1
                    buffer_key = past_key
            else:
                pass
    logging.info(f"Skip ratio is {skip_key_count / total_key_count}")

    return keys

def get_parser():
    parser = argparse.ArgumentParser(
        description="TTT json to text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("json", type=str, help="json files")
    parser.add_argument("dest", type=str, help="output file path")
    parser.add_argument("prep", type=int, help="flag of preprocessing", default=False)
    
    parser.add_argument("total_offset", type=int, help="", default=100)
    parser.add_argument("max_snt_len", type=int, help="", default=150)
    parser.add_argument("max_para_len", type=int, help="", default=1600)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    logging.info("reading %s", args.json)
    with codecs.open(args.json, "r", encoding="utf-8") as f:
        j = json.load(f)

    dest = Path(args.dest)

    # Remove the duplicated keys and load the json to the dict
    prep_j = {}
    for line in tqdm(j):
        try:
            prep_j[line['id']] = {'paragraph': line['paragraph'], 'sentence': line['sentence']}
        except:
            logging.warning("The key %s is duplicated with the exsisted key", line['id'])
    
    # Eliminate the error key with python readlines function
    # FIXME(j-ppng): These lines is fixed by python reading error cleaner.
    # However, we needs to more specific text cleaner
    if args.prep:
        keys = [k for k in prep_j.keys()]

        logging.info("writing train_origin to %s", str(dest))
        train_txt = codecs.open(dest / "text_orig", "w", encoding="utf-8")
        for key in tqdm(keys):
            train_txt.write(key + " " + prep_j[key]['paragraph'] + "\n")

        keys = error_checker(keys, 
                            dest / "text_orig",
                            dest / "error.log")
        
        logging.info("writing key_file to %s", str(dest))
        key_file = codecs.open(dest / "keys", "w", encoding="utf-8")
        for key in keys:
            key_file.write(key + "\n")                            
    else:
        keys = []
        with open(dest / "keys", encoding="utf-8") as f:
            for key in f.readlines():
                keys.append(key.replace("\n", ""))
    
    new_keys = []
    total_offset = args.total_offset
    max_snt_len = args.max_snt_len
    max_para_len = args.max_para_len
    for key in tqdm(keys):
        # find and clipping preprocessing
        # On the first try, we applied these procedures to the middle of the collect_stats process. 
        # However, we found that the {feat}_shape file saves the static size of the features, 
        # and we can know the features shape error will occur when at the training process. 
        idx = prep_j[key]['paragraph'].find(prep_j[key]['sentence'])
        offset = random.randint(0, total_offset)
        sent_len = len(prep_j[key]['sentence'])

        # calculate the offset for the clip with the centroid which sentence in the paragraph.
        prior_offset = max(idx - offset, 0)
        post_offset = idx + sent_len + (total_offset - offset)

        # clip the new paragraph area in the paragraph with the offsets.
        selected_para = prep_j[key]['paragraph'][prior_offset:post_offset]
        para_len = len(selected_para)
        if para_len < sent_len:
            raise RuntimeError(f"prior_offeset: {prior_offset}, post_offset: {post_offset}, length: {para_len}")
        prep_j[key]['paragraph'] = selected_para

        # remove key of the long sentence/paragraph
        if sent_len < max_snt_len and para_len < max_para_len:
            new_keys.append(key)
    logging.info(f"Removed key raio is {1-len(new_keys)/len(keys)}")

    keys = new_keys

    # Save the results
    logging.info("writing train.txt to %s", str(dest))
    train_txt = codecs.open(dest / "text", "w", encoding="utf-8")
    for key in tqdm(keys):
        train_txt.write(prep_j[key]['paragraph'] + "\n")
    
    logging.info("writing train and valid text to %s", str(dest))
    split_point = int(len(keys) * 0.9)
    datasets = {'train': keys[:split_point], 'valid': keys[split_point:]}
    for dataset in datasets.keys():
        logging.info("writing ref trn to %s", str(dest / Path(dataset)))
        input_text = codecs.open(dest / Path(dataset) / "text_input", "w", encoding="utf-8")
        output_text = codecs.open(dest / Path(dataset) / "text_output", "w", encoding="utf-8")

        for key in tqdm(datasets[dataset]):
            input_text.write(key + " " + prep_j[key]['paragraph'] + "\n")
            output_text.write(key + " " + prep_j[key]['sentence'] + "\n")
        
        # If want to check the error of data, just use these lines.
        #  error_checker(keys,
        #             dest / Path(dataset) / "text_input", 
        #             dest / Path(dataset) / "error.log")