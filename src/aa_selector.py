from selector import Selector
from time import time
import numpy as np
from math import log
import random
from scipy.sparse import csr_matrix, lil_matrix

class AASelector(Selector):
    '''
    Selects the test points where the pair of nodes have the highest Adamic/Adar score.
    '''
    def select(self, verbosity=10000):
        t0 = time() # start timer
        A = lil_matrix((self.embeddings.n, self.embeddings.n))
        for i, edge in enumerate(self.embeddings.train):
            v1, v2 = edge
            A[v1.id, v2.id] = 1.
            A[v2.id, v1.id] = 1.
        D = lil_matrix(A.shape)
        for node in self.embeddings.nodes:
            d = len(self.embeddings.neighs[node])
            D[node.id, node.id] = 1 / log(d) if d > 1 else 0
        A = csr_matrix(A)
        pairs_mat = A * D * A
        self.pairs = set()
        train = 0
        v1s, v2s = pairs_mat.nonzero()
        for v1, v2, cn in sorted(zip(v1s, v2s, pairs_mat.data), reverse=True, key=lambda it: it[-1]):
            if v1 == v2:
                continue
            v1, v2 = self.embeddings.id_to_node[v1], self.embeddings.id_to_node[v2]
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                train += 1
                continue
            if (v2, v1) in self.pairs:
                continue
            if v1.name <= v2.name:
                self.pairs.add((v1, v2))
            else:
                self.pairs.add((v2, v1))
            if len(self.pairs) == self.k:
                break

        print((train / 2) / len(self.embeddings.train))
        
        t1 = time() # end timer
        if verbosity > 0:
            print('====== {} pairs selected in {} seconds, {} of them pos. ======'.format(len(self.pairs), t1 - t0, self.true_pos()))
            print('R = {} P = {} % = {}'.format(self.recall(), self.precision(), self.prune_percent()))
        return t1 - t0
