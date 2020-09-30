import numpy as np
import math
from lsh_custom import LSHcustom
from link_waldo_selector import LinkWaldoSelector
from aa_selector import AASelector

class LinkWaldoWithBailoutSelector(LinkWaldoSelector):
    '''
    Selects the test points, with bailout logic implemented.

    Classes that inheret from LinkWaldoWithBailoutSelector should implement a setup() function that returns the number of buckets,
    and a membership(v) function that takes a node as a parameter and returns its membership id.
    '''

    def bailout_augment(self):
        print(len(self.pairs))
        for pair, _ in sorted(self.global_pairs.items(), key=lambda it: it[1]):
            if self.bipartite and pair[0].name[0] == pair[1].name[0]:
                continue
            if len(self.pairs) == self.k - self.bailout_count:
                break
            self.pairs.add(pair)

        if len(self.pairs) < self.k - self.bailout_count:
            for pair, _ in sorted(self.alt_global_pairs.items(), key=lambda it: it[1]):
                if self.bipartite and pair[0].name[0] == pair[1].name[0]:
                    continue
                if len(self.pairs) == self.k - self.bailout_count:
                    break
                self.pairs.add(pair)
        print(len(self.pairs))

        if len(self.pairs) < self.k:
            sel = AASelector('aa', self.G, self.k, self.embeddings, self.output_path, self.seed)
            sel.select(verbosity=0)
            pair_to_aa = dict()
            for v1, v2 in sel.pairs:
                pair_to_aa[(v1, v2)] = sum(1 / math.log(len(self.embeddings.neighs[node])) for node in self.embeddings.neighs[v1].intersection(self.embeddings.neighs[v2]))
            for pair, _ in sorted(pair_to_aa.items(), reverse=True, key=lambda it: it[-1]):
                v1, v2 = pair
                self.pairs.add((v1, v2))
                if len(self.pairs) == self.k:
                    break

    def add_pairs_exact(self, s1, s2, target, stdev):
        pair_sim = dict()
        DistMat = 1 - np.dot(s1.nodeX, s2.nodeX.T)

        seen = set()
        pairs = set()
        seen_train = 0
        global_pairs = dict()
        alt_global_pairs = dict()
        for i, j in zip(*np.unravel_index(np.argsort(DistMat.ravel()), DistMat.shape)):
            v1, v2 = s1.id_to_node[i], s2.id_to_node[j]
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                self.seen_train += 1
                seen_train += 1
                continue
            if (j, i) in seen or v1 == v2:
                continue
            seen.add((i, j))
            if v1.name <= v2.name:
                if len(pairs) < target - stdev:
                    pairs.add((v1, v2))
                elif len(seen) < target + stdev:
                    global_pairs[(v1, v2)] = DistMat[i, j]
                else:
                    alt_global_pairs[(v1, v2)] = DistMat[i, j]
            else:
                if len(pairs) < target - stdev:
                    pairs.add((v2, v1))
                elif len(seen) < target + stdev:
                    global_pairs[(v2, v1)] = DistMat[i, j]
                else:
                    alt_global_pairs[(v2, v1)] = DistMat[i, j]
            if len(seen) == target + stdev + stdev + stdev:
                break
        return pairs, seen_train, global_pairs, alt_global_pairs

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
        seen_train = 0
        global_pairs = dict()
        alt_global_pairs = dict()
        seen_train = 0
        for v1, v2 in lsh.similar_pairs():
            if len(seen) == target + stdev + stdev + stdev:
                break
            v1, v2 = s1.id_to_node[v1], s2.id_to_node[v2]
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                self.seen_train += 1
                seen_train += 1
                continue
            if v1 == v2:
                continue
            seen.add((v1, v2))
            if v1.name <= v2.name:
                if len(pairs) < target - stdev:
                    pairs.add((v1, v2))
                elif len(seen) < target + stdev:
                    global_pairs[(v1, v2)] = 1 - np.dot(v1.emb, v2.emb)
                else:
                    alt_global_pairs[(v1, v2)] = 1 - np.dot(v1.emb, v2.emb)
            else:
                if len(pairs) < target - stdev:
                    pairs.add((v2, v1))
                elif len(seen) < target + stdev:
                    global_pairs[(v2, v1)] = 1 - np.dot(v1.emb, v2.emb)
                else:
                    alt_global_pairs[(v2, v1)] = 1 - np.dot(v1.emb, v2.emb)
        return pairs, seen_train, global_pairs, alt_global_pairs

    def add_pairs_aa(self, s1, s2, target, stdev, equiv_class_size, equiv_class):
        '''
        Add pairs from the equivalence class via the AA proximity model.
        '''
        pairs = set()
        seen = set()
        seen_train = 0
        global_pairs = dict()
        alt_global_pairs = dict()
        for pair, score in sorted(self.equiv_class_to_aa[equiv_class], reverse=True, key=lambda it: it[-1]):
            v1, v2 = pair
            if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                self.seen_train += 1
                seen_train += 1
                continue
            if v1 == v2:
                continue
            seen.add((v1, v2))
            if v1.name <= v2.name:
                if len(pairs) < target - stdev:
                    pairs.add((v1, v2))
                elif len(seen) < target + stdev:
                    global_pairs[(v1, v2)] = 1 / score
                else:
                    alt_global_pairs[(v1, v2)] = 1 / score
            else:
                if len(seen) < target - stdev:
                    pairs.add((v2, v1))
                elif len(seen) < target + stdev:
                    global_pairs[(v2, v1)] = 1 / score
                else:
                    alt_global_pairs[(v2, v1)] = 1 / score
            if len(pairs) == target + stdev + stdev + stdev:
                break
        return pairs, seen_train, global_pairs, alt_global_pairs

    def add_pairs(self, s1, s2, equiv_class_size, target, stdev, equiv_class=None):
        if target - stdev >= equiv_class_size:
            return self.add_all(s1, s2)
        if self.embeddings.name == 'aa':
            res, seen_train, global_pairs, alt_global_pairs = self.add_pairs_aa(s1, s2, target, stdev, equiv_class_size, equiv_class)
        elif equiv_class_size < self.exact_search_tolerance:   
            res, seen_train, global_pairs, alt_global_pairs = self.add_pairs_exact(s1, s2, target, stdev)
        else:
            res, seen_train, global_pairs, alt_global_pairs = self.add_pairs_approx(s1, s2, target, stdev, equiv_class_size)

        if seen_train / round(target / self.k * len(self.embeddings.train)) < self.bailout_tol:
            self.bailout_count += target
            return set()
        for pair, s in global_pairs.items():
            self.global_pairs[pair] = s
        for pair, s in alt_global_pairs.items():
            self.alt_global_pairs[pair] = s
        return res