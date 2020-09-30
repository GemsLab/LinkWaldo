from emb import Emb
import networkx as nx
from karateclub.node_embedding.neighbourhood import NetMF as KarateNetMF
import numpy as np
import random

class NetMF(Emb):
    def run(self, G):
        '''
        :G: a graph to run on.
        '''
        self.G = G
        netmf = KarateNetMF(dimensions=128, order=self.window_size)

        idx_node = dict()
        node_idx = dict()
        for i, node in enumerate(G.nodes()):
            idx_node[i] = node
            node_idx[node] = i
        edges = list()
        for u, v in G.edges():
            edges.append((node_idx[u], node_idx[v]))

        netmf_G = nx.Graph(edges)
        print('Fitting NetMF.')
        netmf.fit(netmf_G)
        embeddings = netmf.get_embedding()
        print('NetMF Fit.')
        i = 0
        with open(self.emb_path, 'w') as f:
            f.write('{} 128\n'.format(len(embeddings)))
            for emb in embeddings:
                f.write('{} {}\n'.format(idx_node[i], ' '.join(list(str(d) for d in emb))))
                i += 1
    
