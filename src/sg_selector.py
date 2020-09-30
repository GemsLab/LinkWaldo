from link_waldo_selector import LinkWaldoSelector
import numpy as np
import os
from xnetmf_emb import XNetMF
from sklearn.cluster import KMeans

class SGSelector(LinkWaldoSelector):
    '''
    Selects the test points ...
    '''

    def membership(self, v):
        return self.clusters[v.id]
        
    def setup(self):
        emb_path = self.embeddings.emb_path.replace(self.embeddings.emb_path.split('/')[-1].split('_')[0], 'xnetmf')
        struc_embeddings = XNetMF(self.G, self.embeddings.edgelist, self.embeddings.test_path, emb_path, self.embeddings.G, normalize=False)
        if not os.path.exists(emb_path):
            struc_embeddings.run(self.embeddings.G)
        struc_embeddings.load_data(load_embeddings=True)

        K = 25 if not self.num_groups else self.num_groups
        print('# Clusters = {}'.format(K))
        self.clusters = KMeans(n_clusters=K, random_state=self.seed).fit_predict(struc_embeddings.nodeX)

        print(np.bincount(self.clusters))
        return K
