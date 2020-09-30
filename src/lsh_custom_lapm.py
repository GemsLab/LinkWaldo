import numpy as np

class LSHcustomLaPM:
    '''
    Uses random hyperplane LSH to find the kapa most similar pairs with high probability.
    '''
    def __init__(self, kapa, num_trees=10, max_depth=12, d=128, verbosity=10000):
        self.verbosity = verbosity
        self.kapa = kapa
        self.target = 5 * kapa
        self.d = d
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X):
        if self.verbosity > 0:
            print('Fitting LSH.')
        self.X = X
        self.trees = list()
        for _ in range(self.num_trees):
            S = set(range(X.shape[0]))
            self.trees.append(self.build_tree(X, S))
        if self.verbosity > 0:
            print('LSH fit.')

    class Tree:
        def __init__(self, X, S, d):
            self.X = X
            self.S = S
            self.d = d
            self.left = None
            self.right = None

        def is_leaf(self):
            return not self.left and not self.right

        def branch(self):
            if self.is_leaf():
                left_S = set()
                right_S = set()
                h = np.random.normal(size=self.d)
                h /= np.linalg.norm(h)
                for i in self.S:
                    v = self.X[i]
                    if np.dot(h, v) >= 0:
                        left_S.add(i)
                    else:
                        right_S.add(i)
                self.left = LSHcustomLaPM.Tree(self.X, left_S, self.d) if len(left_S) > 0 else None
                self.right = LSHcustomLaPM.Tree(self.X, right_S, self.d) if len(right_S) > 0 else None
                assert(self.left or self.right)
                return self.leaf_volume()
            else:
                left_volume, right_volume = 0, 0
                if self.left:
                    left_volume = self.left.branch()
                if self.right:
                    right_volume = self.right.branch()
                return left_volume + right_volume

        def leaf_volume(self):
            if self.is_leaf():
                return len(self.S) ** 2
            elif self.left and self.right:
                return self.left.leaf_volume() + self.right.leaf_volume()
            elif self.left:
                return self.left.leaf_volume()
            elif self.right:
                return self.right.leaf_volume()

        def leaves(self):
            if self.is_leaf():
                return [self.S]
            elif self.left and self.right:
                return self.left.leaves() + self.right.leaves()
            elif self.left:
                return self.left.leaves()
            elif self.right:
                return self.right.leaves()

        def depths(self, depth=0):
            if self.is_leaf():
                return [depth]
            elif self.left and self.right:
                return self.left.depths(depth + 1) + self.right.depths(depth + 1)
            elif self.left:
                return self.left.depths(depth + 1)
            elif self.right:
                return self.right.depths(depth + 1)

        def leaf_sizes(self):
            if self.is_leaf():
                return [len(self.S)]
            elif self.left and self.right:
                return self.left.leaf_sizes() + self.right.leaf_sizes()
            elif self.left:
                return self.left.leaf_sizes()
            elif self.right:
                return self.right.leaf_sizes()


    def build_tree(self, X, S):
        tree = self.Tree(X, S, self.d)
        if self.verbosity > 0:
            print('==========')
        i = 0
        leaf_volume = 10000000000000000000
        while leaf_volume > self.target and leaf_volume > 1000 and i < self.max_depth:
            leaf_volume = tree.branch()
            i += 1
            if self.verbosity > 0:
                print(tree.leaf_volume())
        return tree

    def similar_pairs(self, illegal=set()):
        '''
        Compute the most similar pairs of all those that have been hashed.
        '''
        buckets = list()
        for tree in self.trees:
            buckets.extend(tree.leaves())
        num_buckets = len(buckets)
        print('Num buckets = {}'.format(num_buckets))
        verbosity = 10 ** (int(np.floor(np.log10(num_buckets))) - 1)
        pair_cos = dict()
        for i, bucket in enumerate(buckets):
            for v1 in bucket:
                for v2 in bucket:
                    if v1 == v2 or (v1, v2) in illegal or (v2, v1) in illegal or (v1, v2) in pair_cos or (v2, v1) in pair_cos:
                        continue
                    pair_cos[(v1, v2)] = 1 - np.dot(self.X[v1], self.X[v2])
            if self.verbosity > 0 and i > 0 and i % verbosity == 0:
                print(i / num_buckets)
        
        pairs = set()
        for pair, cos in sorted(pair_cos.items(), key=lambda it: it[-1])[:self.kapa]:
            v1, v2 = pair
            pairs.add((v1, v2))
        return pairs

    def clear(self):
        '''
        Clear the buckets.
        '''
        self.trees = list()
        self.X = None
