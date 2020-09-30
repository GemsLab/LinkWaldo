from selector import Selector
from time import time
import numpy as np
import math
import random
from scipy.sparse import csr_matrix, lil_matrix
from collections import defaultdict
from lsh_custom import LSHcustom
from group import Group

class LinkWaldoSelector(Selector):
    '''
    Selects the test points.

    Classes that inheret from LinkWaldoSelector should implement a setup() function that returns the number of buckets,
    and a membership(v) function that takes a node as a parameter and returns its membership id.
    '''

    def select(self, verbosity=10000):
        '''
        Select k pairs.
        '''
        t0 = time() # start timer

        random.seed(self.seed)
        np.random.seed(self.seed)

        K = self.setup()

        bucket_nodes = defaultdict(set)
        equiv_class_cache = dict()
        for node in self.embeddings.nodes:
            b = self.membership(node)
            equiv_class_cache[node] = b
            bucket_nodes[b].add(node)

        if self.embeddings.name == 'aa':
            self.equiv_class_to_aa = defaultdict(set)
            A = lil_matrix((self.embeddings.n, self.embeddings.n))
            for i, edge in enumerate(self.embeddings.train):
                v1, v2 = edge
                A[v1.id, v2.id] = 1.
                A[v2.id, v1.id] = 1.
            D = lil_matrix(A.shape)
            for node in self.embeddings.nodes:
                d = len(self.embeddings.neighs[node])
                D[node.id, node.id] = 1 / math.log(d) if d > 1 else 0
            A = csr_matrix(A)
            pairs_mat = A * D * A
            t = pairs_mat.count_nonzero()
            print('A^2 {}'.format(t))
            i = 0
            _seen = set()
            for idxs, aa in zip(np.transpose(pairs_mat.nonzero()), pairs_mat.data):
                v1, v2 = idxs
                if v1 == v2 or (v2, v1) in _seen:
                    continue
                _seen.add((v1, v2))
                v1, v2 = self.embeddings.id_to_node[v1], self.embeddings.id_to_node[v2]
                b1 = equiv_class_cache[v1]
                b2 = equiv_class_cache[v2]
                self.equiv_class_to_aa[(b1, b2)].add(((v1, v2), aa))
                self.equiv_class_to_aa[(b2, b1)].add(((v1, v2), aa))
                i += 1
                if verbosity > 0 and i > 0 and i % 1000000 == 0:
                    print(i, i / t)
            print('AA fit.')
        
        # construct road map
        self.road_map = np.zeros((K, K))
        for v1, v2 in self.embeddings.train:
            b1 = self.membership(v1)
            b2 = self.membership(v2)
            self.road_map[b1, b2] += 1.
            if b1 != b2:
                self.road_map[b2, b1] += 1.

        for key in bucket_nodes.keys():
            bucket_nodes[key] = Group(bucket_nodes[key], self.embeddings.d)

        self.pairs = set()
        self.global_pairs = dict()
        self.alt_global_pairs = dict()
        seen = set()
        self.equiv_class_scores = dict()
        for i, j in np.transpose(self.road_map.nonzero()):
            if (j, i) in seen:
                continue
            seen.add((i, j))
            self.equiv_class_scores[(i, j)] = self.road_map[i, j]
        Z = sum(self.equiv_class_scores.values())
        for key in self.equiv_class_scores.keys():
            self.equiv_class_scores[key] /= Z

        self.hash_funcs = None

        i = 0
        self.seen_train = 0
        total_diff = 0
        total = len(self.equiv_class_scores)
        for equiv_class, equiv_class_score in sorted(self.equiv_class_scores.items(), reverse=True, key=lambda it: it[-1]):
            target = max(1, int(round(equiv_class_score * self.k)))
            if target == 0 or len(self.pairs) == self.k:
                continue
            stdev = int(round(np.sqrt(self.k * equiv_class_score * (1 - equiv_class_score))))
            s1 = bucket_nodes[equiv_class[0]]
            s2 = bucket_nodes[equiv_class[1]]
            equiv_class_size = len(s1.nodes) * len(s2.nodes)
            i += 1
            if verbosity > 0 and ((total < 625) or (i > 0 and i % 100 == 0)):
                print((equiv_class[0], equiv_class[1]), i / len(self.equiv_class_scores))
                print('|C| = {}, kapa = {}'.format(equiv_class_size, target))
            res = self.add_pairs(s1, s2, equiv_class_size, target, stdev, equiv_class=equiv_class)
            diff = target - len(res)
            total_diff += diff
            tp = len(res.intersection(self.embeddings.test))
            self.pairs.update(res)
            if verbosity > 0 and ((total < 625) or (i > 0 and i % 100 == 0)):
                print('# pairs = {} with R = {}'.format(len(self.pairs), self.recall()))

        if self.bailout_tol == 0.0:
            for pair, _ in sorted(self.global_pairs.items(), key=lambda it: it[1]):
                if self.bipartite and pair[0].name[0] == pair[1].name[0]:
                    continue
                if len(self.pairs) == self.k:
                    break
                self.pairs.add(pair)

            if len(self.pairs) < self.k:
                for pair, _ in sorted(self.alt_global_pairs.items(), key=lambda it: it[1]):
                    if self.bipartite and pair[0].name[0] == pair[1].name[0]:
                        continue
                    if len(self.pairs) == self.k:
                        break
                    self.pairs.add(pair)
        else:
            self.bailout_augment()


        self.pairs.difference_update(self.embeddings.train)

        t1 = time() # end timer
        if self.skip_output:
            print('====== {} pairs selected in {} seconds ======'.format(len(self.pairs), t1 - t0))
            print('R = {}'.format(self.recall()))
        else:
            print('====== {} pairs selected in {} seconds, {} of them pos. ======'.format(len(self.pairs), t1 - t0, self.true_pos()))
            print('R = {} P = {} % = {}. {} of train encountered'.format(self.recall(), self.precision(), self.prune_percent(), self.seen_train / len(self.embeddings.train)))
        return t1 - t0

    def add_all(self, s1, s2):
        pairs = set()
        for v1 in s1.nodes:
            for v2 in s2.nodes:
                if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                    self.seen_train += 1
                    continue
                if v1 == v2:
                    continue
                if v1.name <= v2.name:
                    pairs.add((v1, v2))
                else:
                    pairs.add((v2, v1))
        return pairs

    def add_pairs_exact(self, s1, s2, target, stdev):
        pair_sim = dict()
        DistMat = 1 - np.dot(s1.nodeX, s2.nodeX.T)

        seen = set()
        pairs = set()
        for i, j in zip(*np.unravel_index(np.argsort(DistMat.ravel()), DistMat.shape)):
            v1, v2 = s1.id_to_node[i], s2.id_to_node[j]
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                self.seen_train += 1
                continue
            if (j, i) in seen or v1 == v2:
                continue
            seen.add((i, j))
            if v1.name <= v2.name:
                if len(pairs) < target - stdev:
                    pairs.add((v1, v2))
                elif len(seen) < target + stdev:
                    self.global_pairs[(v1, v2)] = DistMat[i, j]
                else:
                    self.alt_global_pairs[(v1, v2)] = DistMat[i, j]
            else:
                if len(pairs) < target - stdev:
                    pairs.add((v2, v1))
                elif len(seen) < target + stdev:
                    self.global_pairs[(v2, v1)] = DistMat[i, j]
                else:
                    self.alt_global_pairs[(v2, v1)] = DistMat[i, j]
            if len(seen) == target + stdev + stdev + stdev:
                break
        return pairs

    def add_pairs_approx(self, s1, s2, target, stdev, equiv_class_size):
        if self.hash_funcs is None:
            self.hash_funcs = np.random.normal(size=(10000, self.embeddings.d))
        if s1.hash_codes is None:
            s1.hash(self.hash_funcs)
        if s2.hash_codes is None:
            s2.hash(self.hash_funcs)

        if equiv_class_size < 500000:
            max_depth = 5
        elif equiv_class_size < 750000:
            max_depth = 7
        elif equiv_class_size < 1000000:
            max_depth = 10
        elif equiv_class_size < 1000000000:
            max_depth = 12
        elif equiv_class_size < 10000000000:
            max_depth = 15
        elif equiv_class_size < 25000000000:
            max_depth = 20
        else:
            max_depth = 30
        if target / equiv_class_size < 0.0001:
            num_trees = 5
        elif target / equiv_class_size < 0.001:
            num_trees = 10
        else:
            num_trees = 25
        lsh = LSHcustom(kapa=target, num_trees=num_trees, max_depth=max_depth, d=self.embeddings.d)
        lsh.fit(s1.nodeX, s2.nodeX, s1.hash_codes, s2.hash_codes)
        seen = set()
        pairs = set()
        for v1, v2 in lsh.similar_pairs():
            if len(seen) == target + stdev + stdev + stdev:
                break
            v1, v2 = s1.id_to_node[v1], s2.id_to_node[v2]
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                self.seen_train += 1
                continue
            if v1 == v2:
                continue
            seen.add((v1, v2))
            if v1.name <= v2.name:
                if len(pairs) < target - stdev:
                    pairs.add((v1, v2))
                elif len(seen) < target + stdev:
                    self.global_pairs[(v1, v2)] = 1 - np.dot(v1.emb, v2.emb)
                else:
                    self.alt_global_pairs[(v1, v2)] = 1 - np.dot(v1.emb, v2.emb)
            else:
                if len(pairs) < target - stdev:
                    pairs.add((v2, v1))
                elif len(seen) < target + stdev:
                    self.global_pairs[(v2, v1)] = 1 - np.dot(v1.emb, v2.emb)
                else:
                    self.alt_global_pairs[(v2, v1)] = 1 - np.dot(v1.emb, v2.emb)
        return pairs

    def add_pairs_aa(self, s1, s2, target, stdev, equiv_class_size, equiv_class):
        '''
        Add pairs from the equivalence class via the AA proximity model.
        '''
        pairs = set()
        seen = set()
        for pair, score in sorted(self.equiv_class_to_aa[equiv_class], reverse=True, key=lambda it: it[-1]):
            v1, v2 = pair
            if v1 != v2 and (v1, v2) not in self.embeddings.train and (v2, v1) not in self.embeddings.train:
                seen.add((v1, v2))
                if v1.name <= v2.name:
                    if len(pairs) < target - stdev:
                        pairs.add((v1, v2))
                    elif len(seen) < target + stdev:
                        self.global_pairs[(v1, v2)] = 1 / score
                    else:
                        self.alt_global_pairs[(v1, v2)] = 1 / score
                else:
                    if len(seen) < target - stdev:
                        pairs.add((v2, v1))
                    elif len(seen) < target + stdev:
                        self.global_pairs[(v2, v1)] = 1 / score
                    else:
                        self.alt_global_pairs[(v2, v1)] = 1 / score
            if len(pairs) == target + stdev + stdev + stdev:
                break
        return pairs

    def add_pairs(self, s1, s2, equiv_class_size, target, stdev, equiv_class=None):
        if target - stdev >= equiv_class_size:
            return self.add_all(s1, s2)
        if self.embeddings.name == 'aa':
            return self.add_pairs_aa(s1, s2, target, stdev, equiv_class_size, equiv_class)
        if equiv_class_size < self.exact_search_tolerance:   
            return self.add_pairs_exact(s1, s2, target, stdev)
        else:
            return self.add_pairs_approx(s1, s2, target, stdev, equiv_class_size)