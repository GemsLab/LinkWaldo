from link_waldo_selector import LinkWaldoSelector
import numpy as np
import math

class DGSelector(LinkWaldoSelector):
    '''
    Selects the test points.
    '''
    def membership(self, v):
        d = self.degree[v.name]
        return np.digitize(d, bins=self.bins) - 1

    def setup(self):
        self.degree = dict()
        for node in self.embeddings.nodes:
            self.degree[node.name] = len(self.embeddings.neighs[node])
            
        degree_dist = list(self.degree.values())

        K = 25 if not self.num_groups else self.num_groups
        self.bins = np.logspace(start=0, stop=math.floor(math.log10(max(degree_dist))), num=K, base=10)
        return K