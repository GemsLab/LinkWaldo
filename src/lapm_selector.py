from selector import Selector
from lsh_custom_lapm import LSHcustomLaPM
from lsh_custom import LSHcustom
from time import time
from group import Group

class LaPMSelector(Selector):
    '''
    Selects the test points where the pair of nodes are most similar.
    '''

    def select(self, verbosity=1000):
        t0 = time() # start timer
        self.select_approx(verbosity)
        t1 = time() # end timer
        print('====== {} pairs selected in {} seconds, {} of them pos. ======'.format(len(self.pairs), t1 - t0, self.true_pos()))
        print('R = {} P = {} % = {}'.format(self.recall(), self.precision(), self.prune_percent()))
        return t1 - t0

    def select_approx(self, verbosity):
        if self.bipartite:
            g_nodes = list(self.G.nodes(data='bipartite'))
            print('bip')
            A = set(self.embeddings.node_names_to_nodes[it[0]] for it in filter(lambda it: it[1] == 0, g_nodes))
            B = set(self.embeddings.node_names_to_nodes[it[0]] for it in filter(lambda it: it[1] == 1, g_nodes))
            group_A = Group(A, self.embeddings.d)
            group_B = Group(B, self.embeddings.d)
            lsh = LSHcustom(kapa=self.k, num_trees=25, max_depth=12, d=self.embeddings.d)
            lsh.fit(group_A.nodeX, group_B.nodeX)
            self.pairs = set()
            for v1, v2 in lsh.similar_pairs():
                if len(self.pairs) == self.k:
                    break
                v1, v2 = group_A.id_to_node[v1], group_B.id_to_node[v2]
                if (v1, v2) in self.embeddings.train or (v2, v1) in self.embeddings.train:
                    continue
                if v1 == v2:
                    continue
                if v1.name <= v2.name:
                    self.pairs.add((v1, v2))
                else:
                    self.pairs.add((v2, v1))
            return

        lsh = LSHcustomLaPM(self.k, num_trees=25, d=self.embeddings.d)
        lsh.fit(self.embeddings.nodeX)
        illegal = set((v1.id, v2.id) for v1, v2 in self.embeddings.train)
        self.pairs = set(tuple(sorted((self.embeddings.id_to_node[v1], self.embeddings.id_to_node[v2]), key=lambda it: it.name)) for v1, v2 in lsh.similar_pairs(illegal=illegal))