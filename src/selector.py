import numpy as np

class Selector:
    '''
    Selects the test points that look most like a training point.
    '''
    def __init__(self, name, G, k, embeddings, output_path, exact_search_tolerance=25000000, bipartite=False, DG=True, SG=True, CG=True, seed=0):
        self.name = name
        self.G = G
        self.k = k
        self.embeddings = embeddings
        self.output_path = output_path
        self.exact_search_tolerance = exact_search_tolerance
        self.seed = 0
        self.bipartite = bipartite
        self.DG = DG
        self.SG = SG
        self.CG = CG
        self.num_groups = None
        self.num_groups_alt = None
        self.bailout_tol = None
        self.bag_epsilon = None
        self.skip_output = False
        print('k = {}'.format(self.k))

    def true_pos(self):
        return len(self.pairs.intersection(self.embeddings.test))

    def recall(self):
        return len(self.pairs.intersection(self.embeddings.test)) / len(self.embeddings.test)

    def precision(self):
        return len(self.pairs.intersection(self.embeddings.test)) / len(self.pairs)

    def prune_percent(self):
        return len(self.pairs) / self.embeddings.n ** 2

    def write(self):
        with open(self.output_path, 'w') as f:
            for v1, v2 in self.pairs:
                f.write('{} {}\n'.format(v1.name, v2.name))

    def write_res(self, t):
        with open(self.output_path.replace('.txt', '_num.txt'), 'w') as f:
            f.write('{}\n'.format(len(self.pairs)))
        with open(self.output_path.replace('.txt', '_R.txt'), 'w') as f:
            f.write('{}\n'.format(self.recall()))
        with open(self.output_path.replace('.txt', '_P.txt'), 'w') as f:
            f.write('{}\n'.format(self.precision()))
        with open(self.output_path.replace('.txt', '_frac.txt'), 'w') as f:
            f.write('{}\n'.format(self.prune_percent()))
        with open(self.output_path.replace('.txt', '_time.txt'), 'w') as f:
            f.write('{}\n'.format(t))
