from heapq import heappush, nsmallest
import numpy as np

class NearestNeighbor():
    def __init__(self, embeddings, encodings, config):
        self.embeddings = embeddings
        self.encodings = encodings
        self.config = config

    def euclidian_distance(self, e1, e2):
        '''
        https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
        '''
        return np.linalg.norm(e1 - e2)

    def get_embedding(self, word):
        if self.encodings.word_in_vocab(word):
            return self.embeddings[word]

        return self.embeddings[config.unknown_word]

    def nearest_neighbors(self, word, count=1):
        embedding = self.get_embedding(word)
        heap = []

        # TODO: is it faster to not have the the string comparision and instead always
        #       remove the first element of the array which will have a distance of 0
        # TODO: implement faster solution than the heap where it only keeps track of K
        #       values which should vastly reduce the number of operations required.
        for w in self.embeddings:
            if w == word:
                continue

            dist = self.euclidian_distance(embedding, self.embeddings[w])
            heappush(heap, (dist, w))

        return nsmallest(count, heap)
