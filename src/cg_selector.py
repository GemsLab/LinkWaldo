from link_waldo_selector import LinkWaldoSelector
import numpy as np
from sklearn.cluster import KMeans

class CGSelector(LinkWaldoSelector):
    '''
    Selects the test points ...
    '''

    def membership(self, v):
        return self.clusters[v.id]
        
    def setup(self):
        K = 25 if not self.num_groups else self.num_groups
        print('# Clusters = {}'.format(K))
        self.clusters = KMeans(n_clusters=K, random_state=self.seed).fit_predict(self.embeddings.nodeX)

        print(np.bincount(self.clusters))
        return K
