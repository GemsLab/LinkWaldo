import numpy as np

class Group:
    def __init__(self, s, d):
        self.nodes = s
        self.id_to_node = dict()
        self.nodeX = np.zeros((len(s), d))
        for i, node in enumerate(s):
            self.nodeX[i] = node.emb
            self.id_to_node[i] = node
        self.hash_codes = None

    def hash(self, H):
        self.hash_codes = np.sign(np.matmul(H, self.nodeX.T).T)