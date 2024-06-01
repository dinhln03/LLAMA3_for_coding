import argparse
from pathlib import Path
import tempfile
from typing import List

from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from target_extraction.data_types import TargetTextCollection
from target_extraction.tokenizers import spacy_tokenizer

def parse_path(path_string: str) -> Path:
    path_string = Path(path_string).resolve()
    return path_string

def shrink_glove_file(glove_fp: Path, filter_words: List[str], save_fp: Path
                      ) -> None:
    '''
    :param glove_fp: File path to the glove file that is to be shrinked
    :param filter_words: List of words to filter/shrink the glove file/vectors 
                         by
    :param save_fp:
    '''
    with save_fp.open('w+') as save_file:
        with glove_fp.open('r') as glove_file:
            for glove_vector in glove_file:
                glove_parts = glove_vector.split()
                if (len(glove_parts) == 301 or len(glove_parts) == 51 or 
                    len(glove_parts) == 201):
                    pass
                else:
                    continue
                glove_word = glove_parts[0]
                if glove_word in filter_words:
                    save_file.write(glove_vector)

#python tdsa_augmentation/data_creation/shrink_glove_to_targets.py ./data/original_restaurant_sentiment/train.json ./resources/word_embeddings/glove.840B.300d.txt ./here
if __name__ == '__main__':
    glove_fp_help = 'File path to the Glove embedding to be shrunk and '\
                    'converted to Word2Vec format'
    parser = argparse.ArgumentParser()
    parser.add_argument("json_train_data", type=parse_path, 
                        help='File path JSON training data')
    parser.add_argument("glove_embedding_fp", type=parse_path, 
                        help=glove_fp_help)
    parser.add_argument("target_only_word2vec_path", type=parse_path, 
                        help='File path to save the embedding too.')
    args = parser.parse_args()

    save_fp = args.target_only_word2vec_path
    if save_fp.exists():
        print('A file already exists at the location to store '
              f'the new Word2Vec model/vector: {save_fp}\n'
              'Thus skipping the rest of this script.')
    else:
        dataset = TargetTextCollection.load_json(args.json_train_data)
        all_targets = list(dataset.target_count(lower=True).keys())
        tokenizer = spacy_tokenizer()
        tokenised_targets = [target for targets in all_targets for target in tokenizer(targets)]
        with tempfile.TemporaryDirectory() as temp_dir:
            shrink_glove_temp_fp = Path(temp_dir, 'temp_glove')
            shrink_word_vec_temp_fp = Path(temp_dir, 'temp_wordvec')
            shrink_glove_file(args.glove_embedding_fp, tokenised_targets, shrink_glove_temp_fp)
            glove2word2vec(shrink_glove_temp_fp, shrink_word_vec_temp_fp)
            
            model = KeyedVectors.load_word2vec_format(shrink_word_vec_temp_fp)
            model.save(str(save_fp))
        print(f'Word2Vec shrunk to target model saved to {save_fp}')

            



