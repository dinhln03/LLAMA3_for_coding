import nltk
import numpy as np
#nltk.download('punkt')       #downloading a package with a pretrained tokenizer
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentence):      #splitting a string into meaningful units
    return nltk.word_tokenize(sentence)

def stem(word):              #Generating the root form of the words
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_words), dtype = np.float32)
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag




