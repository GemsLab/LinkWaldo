import math
import numpy as np
import random

class LSHcustom:
    '''
    Uses random hyperplane LSH to find the kappa most similar pairs with high probability.
    '''
    def __init__(self, kappa, num_trees=10, max_depth=12, d=128, verbosity=10000):
        self.verbosity = verbosity
        self.target = 5 * kappa
        self.d = d
        self.num_trees = num_trees
        self.max_depth = max_depth     

    def fit(self, X1, X2, hash_codes1=None, hash_codes2=None):
        if self.verbosity > 0:
            print('Fitting LSH {}.'.format(X1.shape[0] * X2.shape[0]))
        self.X1 = X1
        self.X2 = X2

        self.hash_codes1 = hash_codes1
        self.hash_codes2 = hash_codes2
        if not hash_codes1 is None:
            self.hash_funcs = list(range(hash_codes1.shape[1]))
            random.shuffle(self.hash_funcs)
        else:
            self.hash_funcs = list()       

        self.n = X1.shape[0] + X2.shape[0]
        self.trees = list()
        hash_funcs = self.hash_funcs
        for _ in range(self.num_trees):
            S1 = set(range(X1.shape[0]))
            S2 = set(range(X2.shape[0]))
            self.trees.append(self.build_tree(X1, X2, S1, S2, hash_codes1, hash_codes2, hash_funcs))
            hash_funcs = self.trees[-1].hash_funcs
        if self.verbosity > 0:
            print('LSH fit.')

    class Tree:
        def __init__(self, X1, X2, S1, S2, d, hash_codes1, hash_codes2, hash_funcs):
            self.X1 = X1
            self.X2 = X2
            self.S1 = S1
            self.S2 = S2
            self.d = d
            self.left = None
            self.right = None
            self.hash_codes1 = hash_codes1
            self.hash_codes2 = hash_codes2
            self.hash_funcs = hash_funcs

        def is_leaf(self):
            return not self.left and not self.right

        def branch(self):
            if self.is_leaf():
                left_S1 = set()
                left_S2 = set()
                right_S1 = set()
                right_S2 = set()

                if len(self.hash_funcs) == 0:
                    h = np.random.normal(size=self.d)
                    h /= np.linalg.norm(h)
                    for i in self.S1:
                        v = self.X1[i]
                        if np.dot(h, v) >= 0:
                            left_S1.add(i)
                        else:
                            right_S1.add(i)

                    for i in self.S2:
                        v = self.X2[i]
                        if np.dot(h, v) >= 0:
                            left_S2.add(i)
                        else:
                            right_S2.add(i)
                else:
                    h = self.hash_funcs.pop()
                    codes1 = self.hash_codes1[:,h]
                    codes2 = self.hash_codes2[:,h]
                    for i in self.S1:
                        if codes1[i] >= 0:
                            left_S1.add(i)
                        else:
                            right_S1.add(i)

                    for i in self.S2:
                        if codes2[i] >= 0:
                            left_S2.add(i)
                        else:
                            right_S2.add(i)

                self.left = LSHcustom.Tree(self.X1, self.X2, left_S1, left_S2, self.d, self.hash_codes1, self.hash_codes2, self.hash_funcs) if len(left_S1) > 0 or len(left_S2) > 0 else None
                self.right = LSHcustom.Tree(self.X1, self.X2, right_S1, right_S2, self.d, self.hash_codes1, self.hash_codes2, self.hash_funcs) if len(right_S1) > 0 or len(right_S2) > 0 else None
                return self.leaf_volume()
            else:
                left_volume, right_volume = 0, 0
                if self.left:
                    left_volume = self.left.branch()
                if self.right:
                    right_volume = self.right.branch()
                return left_volume + right_volume

        def unbranch(self):
            if (not self.left or self.left.is_leaf()) and (not self.right or self.right.is_leaf()):
                self.left = None
                self.right = None
            else:
                if self.left:
                    self.left.unbranch()
                if self.right:
                    self.right.unbranch()

        def leaf_volume(self):
            if self.is_leaf():
                return len(self.S1) * len(self.S2)
            elif self.left and self.right:
                return self.left.leaf_volume() + self.right.leaf_volume()
            elif self.left:
                return self.left.leaf_volume()
            elif self.right:
                return self.right.leaf_volume()

        def leaves(self):
            if self.is_leaf():
                return [(self.S1, self.S2)]
            elif self.left and self.right:
                return self.left.leaves() + self.right.leaves()
            elif self.left:
                return self.left.leaves()
            elif self.right:
                return self.right.leaves()

        def leaf_depths(self, depth):
            if self.is_leaf():
                return [depth]
            elif self.left and self.right:
                return self.left.leaf_depths(depth + 1) + self.right.leaf_depths(depth + 1)
            elif self.left:
                return self.left.leaf_depths(depth + 1)
            elif self.right:
                return self.right.leaf_depths(depth + 1)

    def build_tree(self, X1, X2, S1, S2, hash_codes1=None, hash_codes2=None, hash_funcs=list()):
        tree = self.Tree(X1, X2, S1, S2, self.d, hash_codes1, hash_codes2, hash_funcs)
        i = 0
        leaf_volume = 10000000000000000000
        while leaf_volume > self.target and leaf_volume > 100000 and i < self.max_depth:
            leaf_volume = tree.branch()
            if leaf_volume < math.ceil(self.target / 5):
                print('Unbranching.')
                tree.unbranch()
            i += 1
        return tree

    def similar_pairs(self, illegal=False):
        '''
        Compute the most similar pairs of all those that have been hashed.
        '''
        buckets = list()
        for tree in self.trees:
            buckets.extend(tree.leaves())
        num_buckets = len(buckets)
        if self.verbosity > 0:
            print('Num buckets = {}'.format(num_buckets))
        verbosity = 10 ** (int(np.floor(np.log10(num_buckets))) - 1)
        pair_cos = dict()
        for i, _buckets in enumerate(buckets):
            bucket1, bucket2 = _buckets
            for v1 in bucket1:
                for v2 in bucket2:
                    if (illegal and (v1, v2) in illegal) or (v1, v2) in pair_cos or (v2, v1) in pair_cos:
                        continue
                    pair_cos[(v1, v2)] = 1 - np.dot(self.X1[v1], self.X2[v2])

            if num_buckets == self.num_trees:
                break
            if self.verbosity > 0 and self.n > 1000000 and i > 0 and i % verbosity == 0:
                print(i / num_buckets)
        
        res = list(it[0] for it in sorted(pair_cos.items(), key=lambda it: it[-1]))
        if self.verbosity > 0:
            print('Pairs found.')
        return res

    def clear(self):
        '''
        Clear the buckets.
        '''
        self.trees = list()
        self.X = None
