from selector import Selector
from time import time
import numpy as np
import random
from scipy.sparse import csr_matrix, lil_matrix

class JSSelector(Selector):
    '''
    Selects the test points where the pair of nodes have the highest Jaccard Similarity.
    '''
    def select(self, verbosity=10000):
        t0 = time() # start timer
        A = lil_matrix((self.embeddings.n, self.embeddings.n))
        for i, edge in enumerate(self.embeddings.train):
            v1, v2 = edge
            A[v1.id, v2.id] = 1.
            A[v2.id, v1.id] = 1.
        A = csr_matrix(A)
        pairs_mat = A * A
        self.pairs = set()
        pair_to_js = dict()
        for v1, v2 in zip(*pairs_mat.nonzero()):
            if v1 == v2:
                continue
            # cn = pairs_mat[v1, v2]
            v1, v2 = self.embeddings.id_to_node[v1], self.embeddings.id_to_node[v2]
            if (v2, v1) in pair_to_js or (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                continue
            pair_to_js[(v1, v2)] = len(self.embeddings.neighs[v1].intersection(self.embeddings.neighs[v2])) / len(self.embeddings.neighs[v1].union(self.embeddings.neighs[v2]))
        for pair, _ in sorted(pair_to_js.items(), reverse=True, key=lambda it: it[-1])[:self.k]:
            v1, v2 = pair
            if v1.name <= v2.name:
                self.pairs.add((v1, v2))
            else:
                self.pairs.add((v2, v1))
        
        t1 = time() # end timer
        print('====== {} pairs selected in {} seconds, {} of them pos. ======'.format(len(self.pairs), t1 - t0, self.true_pos()))
        print('R = {} P = {} % = {}'.format(self.recall(), self.precision(), self.prune_percent()))
        return t1 - t0
