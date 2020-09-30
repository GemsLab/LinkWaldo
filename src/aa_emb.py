from emb import Emb
import networkx as nx
import numpy as np
import random

class AA(Emb):
    def run(self, G):
        '''
        :G: a graph to run on.
        '''
        self.G = G

        with open(self.emb_path, 'w') as f:
            f.write('{} 0\n'.format(G.number_of_nodes()))
            for node in G.nodes():
                f.write('{}\n'.format(node))
    
