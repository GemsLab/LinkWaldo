from node import Node
import numpy as np
import random
import os
from collections import defaultdict
from sklearn.preprocessing import normalize

class Emb:
    '''
    A super class for node embedding methods. Subclasses implement the run() method, which generates embeddings.
    '''
    def __init__(self, name, edgelist, test_path, emb_path, G, normalize, window_size=None):
        self.edgelist = edgelist
        self.test_path = test_path
        self.emb_path = emb_path
        self.name = name
        self.G = G
        self.normalize = normalize
        self.window_size = window_size # for NetMF only

        self.node_emb = dict()
        self.node_names_to_nodes = dict()
        self.nodes = set()
        self.id_to_node = dict()

    def contains(self, node_name):
        return node_name in self.node_names_to_nodes

    def check(self, u, v):
        return self.contains(u) and self.contains(v)

    def node(self, idx):
        return self.node_names_to_nodes[idx]

    def load_embeddings_only(self):
        print('Loading data.')
        nodeX_path = self.emb_path.replace('.emb', '_nodeX.npy')
        if os.path.exists(nodeX_path):
            self.nodeX = np.load(nodeX_path)
            if self.normalize:
                self.nodeX = normalize(self.nodeX)
        else:
            idx = 0
            for line in open(self.emb_path, 'r'):
                line = line.strip().split()
                if len(line) == 2:
                    continue
                node, emb = line[0], np.asarray(list(float(d) for d in line[1:]))
                if np.count_nonzero(np.isinf(emb)) > 0:
                    print('Inf encountered. Replacing with random vector.')
                    emb = np.random.normal(size=len(emb))
                if self.normalize:
                    emb = emb / np.linalg.norm(emb)
                node = Node(node, idx, emb)
                self.nodes.add(node)
                idx += 1            

            self.n = len(self.nodes)
            self.d = len(emb)
            self.nodeX = np.zeros((self.n, self.d))
            for node in self.nodes:
                self.nodeX[node.id] = node.emb
            np.save(nodeX_path, self.nodeX)
        print('Data loaded.')

    def load_data(self, load_embeddings):
        print('Loading data.')
        nodeX_path = self.emb_path.replace('.emb', '_nodeX.npy')
        if os.path.exists(nodeX_path):
            self.nodeX = np.load(nodeX_path)
        else:
            self.nodeX = None

        idx = 0
        for line in open(self.emb_path, 'r'):
            line = line.strip().split()
            if len(line) == 2:
                continue
            if load_embeddings:
                node = line[0]
                if self.nodeX is None:
                    emb = np.asarray(list(float(d) for d in line[1:]))
                else:
                    emb = self.nodeX[idx]
                if self.normalize:
                    emb = emb / np.linalg.norm(emb)
                self.node_names_to_nodes[node] = Node(node, idx, emb)
            else:
                node = line[0]
                self.node_names_to_nodes[node] = Node(node, idx, None)
            self.nodes.add(self.node_names_to_nodes[node])
            self.id_to_node[idx] = self.node_names_to_nodes[node]
            idx += 1            

        self.n = len(self.nodes)

        if load_embeddings:
            self.d = len(emb)
            if self.nodeX is None:
                self.nodeX = np.zeros((self.n, self.d))
                for node in self.nodes:
                    self.nodeX[node.id] = node.emb
        else:
            self.d = 1

        self.train = set()
        self.neighs = defaultdict(set)
        for line in open(self.edgelist, 'r'):
            u, v, *rest = line.strip().split()
            u, v = self.node(u), self.node(v)
            if u.name <= v.name:
                self.train.add((u, v))
            else:
                self.train.add((v, u))
            self.neighs[u].add(v)
            self.neighs[v].add(u)

        self.test = set()
        for line in open(self.test_path, 'r'):
            u, v, *rest = line.strip().split()
            if not self.check(u, v):
                continue
            u, v = self.node(u), self.node(v)
            if u.name <= v.name:
                self.test.add((u, v))
            else:
                self.test.add((v, u))

        print('Train size: {} Test size: {}'.format(len(self.train), len(self.test)))

        if load_embeddings:
            if not os.path.exists(nodeX_path):
                np.save(nodeX_path, self.nodeX)

        print('Data loaded.')
