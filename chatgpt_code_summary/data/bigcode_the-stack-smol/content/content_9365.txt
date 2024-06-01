from dataloader import AmazonProductDataloader
from inverted_index import InvertedIndex
from utils import preprocess_text
import numpy as np

class BM25SearchRelevance:
    def __init__(self, inverted_index, b=0.65, k1=1.6):
        self.inverted_index = inverted_index
        self.b = b
        self.k1 = k1
        self.total_documents = inverted_index.dataloader.dataset.shape[0]
        self.total_documents

    def score_query(self, query, k=3):
        scores = {}
        preprocessed_query = preprocess_text(query, tokens_only=True)
        for query_term in preprocessed_query:
            if query_term in self.inverted_index.term_dictionary:
                term_frequencies = self.inverted_index.term_dictionary[query_term]
                for term_frequency in term_frequencies:
                    if term_frequency["document"] not in scores:
                        scores[term_frequency["document"]] = 0
                    scores[term_frequency["document"]] += self.bm25_score(term_frequency["frequency"], len(term_frequency), term_frequency["document_length"])

        scores = dict(sorted(sorted(scores.items(), key=lambda x: x[1])))
        if k > len(scores.keys()):
            k = len(scores.keys())
        return list(scores.keys())[:k] ## returns top k documents

    def bm25_score(self, term_frequency, document_frequency, document_length):
        tf = term_frequency / self.k1 * ((1-self.b) + (self.b * (document_length / self.inverted_index.average_document_length))) + term_frequency
        idf = np.log((self.total_documents - document_frequency + 0.5)/ (document_frequency + 0.5))
        return tf * idf

