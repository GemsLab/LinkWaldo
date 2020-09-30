class Node:
    def __init__(self, name, idx, emb):
        '''
        A node object.

        :name: is the name of the node *in the graph*
        :idx: is the id of the node *in an embedding matrix*
        :emb: is a vector representation of the node
        '''
        self.name = name
        self.id = idx
        self.emb = emb
