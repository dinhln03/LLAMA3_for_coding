import os
import glob
import json
import argparse

from tokenizers import ByteLevelBPETokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_path', type=str)
    parser.add_argument('--n_files', type=int)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--vocab_size', type=int)
    parser.add_argument('--control_codes', nargs='+',
                        default=['<|endoftext|>'])

    args = parser.parse_args()

    if os.path.isdir(args.train_path):
        paths = glob.glob(os.path.join(args.train_path, '*'))
    else:
        paths = [args.train_path]

    paths = paths[:args.n_files]

    tok = ByteLevelBPETokenizer()

    tok.train(files=paths, vocab_size=args.vocab_size,
              special_tokens=args.control_codes)

    tok.save(args.save_path)

    tokenizer_config = {
        "max_len": 1024
    }

    with open(os.path.join(args.save_path, "tokenizer_config.json"), 'w') as fp:
        json.dump(tokenizer_config, fp)
