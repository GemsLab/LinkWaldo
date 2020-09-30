from link_waldo_with_bailout_selector import LinkWaldoWithBailoutSelector
import numpy as np
import math
import os
import random
from xnetmf_emb import XNetMF
from netmf_emb import NetMF
from sklearn.cluster import KMeans
from group import Group

class MGBailoutSelector(LinkWaldoWithBailoutSelector):
    '''
    '''
    def membership(self, v):
        if self.bins is not None and self.xnetmf_clusters is not None and self.clusters is not None:
            d = self.degree[v.name]
            b1 = np.digitize(d, bins=self.bins) - 1
            b2 = self.xnetmf_clusters[v.id]
            b3 = self.clusters[v.id]
            if self.bipartite:
                b4 = self.bipartite_id[v.name]
                return self.buckets[(b1, b2, b3, b4)]
            return self.buckets[(b1, b2, b3)]
        elif self.bins is not None and self.xnetmf_clusters is not None:
            d = self.degree[v.name]
            b1 = np.digitize(d, bins=self.bins) - 1
            b2 = self.xnetmf_clusters[v.id]
            if self.bipartite:
                b4 = self.bipartite_id[v.name]
                return self.buckets[(b1, b2, b4)]
            return self.buckets[(b1, b2)]
        elif self.bins is not None and self.clusters is not None:
            d = self.degree[v.name]
            b1 = np.digitize(d, bins=self.bins) - 1
            b3 = self.clusters[v.id]
            if self.bipartite:
                b4 = self.bipartite_id[v.name]
                return self.buckets[(b1, b3, b4)]
            return self.buckets[(b1, b3)]
        elif self.xnetmf_clusters is not None and self.clusters is not None:
            b2 = self.xnetmf_clusters[v.id]
            b3 = self.clusters[v.id]
            if self.bipartite:
                b4 = self.bipartite_id[v.name]
                return self.buckets[(b2, b3, b4)]
            return self.buckets[(b2, b3)]            

    def setup(self):
        self.bailout_count = 0
        K1 = 0
        self.bins = None
        if self.DG:
            '''
            setup log-binned degree
            '''
            self.degree = dict()
            for node in self.embeddings.nodes:
                self.degree[node.name] = len(self.embeddings.neighs[node])
            
            degree_dist = list(self.degree.values())
            K1 = 25 if not self.num_groups else self.num_groups
            self.bins = np.logspace(start=0, stop=math.floor(math.log10(max(degree_dist))), num=K1, base=10)

        K2 = 0
        self.xnetmf_clusters = None
        if self.SG:
            '''
            setup xNetMF structural embeddings
            '''
            emb_path = self.embeddings.emb_path.replace(self.embeddings.name, 'xnetmf')
            struc_embeddings = XNetMF(self.G, self.embeddings.edgelist, self.embeddings.test_path, emb_path, self.embeddings.G, normalize=False)
            if not os.path.exists(emb_path):
                struc_embeddings.run(self.embeddings.G)
            struc_embeddings.load_embeddings_only()

            K2 = 5 if not self.num_groups_alt else self.num_groups_alt
            self.xnetmf_clusters = KMeans(n_clusters=K2, random_state=self.seed).fit_predict(struc_embeddings.nodeX)

            print(np.bincount(self.xnetmf_clusters))

        K3 = 0
        self.clusters = None
        if self.CG:
            '''
            setup cluster
            '''
            K3 = 5 if not self.num_groups_alt else self.num_groups_alt
            if self.embeddings.name != 'netmf1' and self.embeddings.name != 'bine':
                emb_path = self.embeddings.emb_path.replace(self.embeddings.name, 'netmf1')
                clus_embeddings = NetMF(self.G, self.embeddings.edgelist, self.embeddings.test_path, emb_path, self.embeddings.G, normalize=False, window_size=1)
                if not os.path.exists(emb_path):
                    clus_embeddings.run(self.embeddings.G)
                clus_embeddings.load_embeddings_only()
                self.clusters = KMeans(n_clusters=K3, random_state=self.seed).fit_predict(clus_embeddings.nodeX)
            else:
                self.clusters = KMeans(n_clusters=K3, random_state=self.seed).fit_predict(self.embeddings.nodeX)

            print(np.bincount(self.clusters))

        '''
        setup Bipartite
        '''
        if self.bipartite:
            K4 = 2
            self.bipartite_id = dict()
            for u, v in self.G.edges():
                self.bipartite_id[u] = 0
                self.bipartite_id[v] = 1
            self.buckets = dict()
            idx = 0
            for i in range(K1):
                for j in range(K2):
                    for k in range(K3):
                        for l in range(K4):
                            if (i, j, k, l) not in self.buckets:
                                self.buckets[(i, j, k, l)] = idx
                                idx += 1
            return K1 * K2 * K3 * K4

        print(K1, K2, K3)

        '''
        setup combo
        '''
        self.buckets = dict()
        idx = 0
        if K1 and K2 and K3:
            for i in range(K1):
                for j in range(K2):
                    for k in range(K3):
                        if (i, j, k) not in self.buckets:
                            self.buckets[(i, j, k)] = idx
                            idx += 1
            return K1 * K2 * K3
        elif K1 and K2:
            for i in range(K1):
                for j in range(K2):
                    if (i, j) not in self.buckets:
                        self.buckets[(i, j)] = idx
                        idx += 1
            return K1 * K2
        elif K1 and K3:
            for i in range(K1):
                for j in range(K3):
                    if (i, j) not in self.buckets:
                        self.buckets[(i, j)] = idx
                        idx += 1
            return K1 * K3
        elif K2 and K3:
            for j in range(K2):
                for k in range(K3):
                    if (j, k) not in self.buckets:
                        self.buckets[(j, k)] = idx
                        idx += 1
            return K2 * K3
        assert(False)