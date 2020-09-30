import scipy as sp
import networkx as nx
import math, time, os, sys
from scipy.spatial.distance import cosine
from emb import Emb
import numpy as np
import random

class XNetMF(Emb):
    def run(self, G):
        '''
        :G: a graph to run on.
        '''
        self.G = G

        print('Fitting xNetMF.')
        graph = self.Graph(nx.adjacency_matrix(self.G))
        rep_method = self.RepMethod(max_layer = 2) #Learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
        representations = self.get_representations(graph, rep_method, verbose=False)
        with open(self.emb_path, 'w') as f:
            f.write('{} {}\n'.format(representations.shape[0], representations.shape[1]))
            for node, emb in zip(self.G.nodes(), representations):
                str_emb = ' '.join(list(str(d) for d in emb))
                f.write('{} {}\n'.format(node, str_emb))
        print('xNetMF Fit.')

    class RepMethod():
        def __init__(self,
                    align_info = None,
                    p=None,
                    k=10,
                    max_layer=None,
                    alpha = 0.1,
                    num_buckets = None,
                    normalize = True,
                    gammastruc = 1,
                    gammaattr = 1):
            self.p = p #sample p points
            self.k = k #control sample size
            self.max_layer = max_layer #furthest hop distance up to which to compare neighbors
            self.alpha = alpha #discount factor for higher layers
            self.num_buckets = num_buckets #number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
            self.normalize = normalize #whether to normalize node embeddings
            self.gammastruc = gammastruc #parameter weighing structural similarity in node identity
            self.gammaattr = gammaattr #parameter weighing attribute similarity in node identity

    class Graph():
        #Undirected, unweighted
        def __init__(self,
                     adj,
                     num_buckets=None,
                     node_labels=None,
                     edge_labels=None,
                     graph_label=None,
                     node_attributes=None,
                     true_alignments=None):
            self.G_adj = adj #adjacency matrix
            self.N = self.G_adj.shape[0] #number of nodes
            self.node_degrees = np.ravel(np.sum(self.G_adj, axis=0).astype(int))
            self.max_degree = max(self.node_degrees)
            self.num_buckets = num_buckets #how many buckets to break node features into

            self.node_labels = node_labels
            self.edge_labels = edge_labels
            self.graph_label = graph_label
            self.node_attributes = node_attributes #N x A matrix, where N is # of nodes, and A is # of attributes
            self.kneighbors = None #dict of k-hop neighbors for each node
            self.true_alignments = true_alignments #dict of true alignments, if this graph is a combination of multiple graphs

    #Input: graph, RepMethod
    #Output: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
    #        dictionary {node ID: degree}
    def get_khop_neighbors(self, graph, rep_method):
        if rep_method.max_layer is None:
            rep_method.max_layer = graph.N #Don't need this line, just sanity prevent infinite loop

        kneighbors_dict = {}

        #only 0-hop neighbor of a node is itself
        #neighbors of a node have nonzero connections to it in adj matrix
        for node in range(graph.N):
            neighbors = np.nonzero(graph.G_adj[node])[-1].tolist() ###
            if len(neighbors) == 0: #disconnected node
                print("Warning: node %d is disconnected" % node)
                kneighbors_dict[node] = {0: set([node]), 1: set()}
            else:
                if type(neighbors[0]) is list:
                    neighbors = neighbors[0]
                kneighbors_dict[node] = {0: set([node]), 1: set(neighbors) - set([node]) }

        #For each node, keep track of neighbors we've already seen
        all_neighbors = {}
        for node in range(graph.N):
            all_neighbors[node] = set([node])
            all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])

        #Recursively compute neighbors in k
        #Neighbors of k-1 hop neighbors, unless we've already seen them before
        current_layer = 2 #need to at least consider neighbors
        while True:
            if rep_method.max_layer is not None and current_layer > rep_method.max_layer: break
            reached_max_layer = True #whether we've reached the graph diameter

            for i in range(graph.N):
                #All neighbors k-1 hops away
                neighbors_prevhop = kneighbors_dict[i][current_layer - 1]

                khop_neighbors = set()
                #Add neighbors of each k-1 hop neighbors
                for n in neighbors_prevhop:
                    neighbors_of_n = kneighbors_dict[n][1]
                    for neighbor2nd in neighbors_of_n:
                        khop_neighbors.add(neighbor2nd)

                #Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
                khop_neighbors = khop_neighbors - all_neighbors[i]

                #Add neighbors at this hop to set of nodes we've already seen
                num_nodes_seen_before = len(all_neighbors[i])
                all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
                num_nodes_seen_after = len(all_neighbors[i])

                #See if we've added any more neighbors
                #If so, we may not have reached the max layer: we have to see if these nodes have neighbors
                if len(khop_neighbors) > 0:
                    reached_max_layer = False

                #add neighbors
                kneighbors_dict[i][current_layer] = khop_neighbors #k-hop neighbors must be at least k hops away

            if reached_max_layer:
                break #finished finding neighborhoods (to the depth that we want)
            else:
                current_layer += 1 #move out to next layer

        return kneighbors_dict


    #Turn lists of neighbors into a degree sequence
    #Input: graph, RepMethod, node's neighbors at a given layer, the node
    #Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
    def get_degree_sequence(self, graph, rep_method, kneighbors, current_node):
        if rep_method.num_buckets is not None:
            degree_counts = [0] * int(math.log(graph.max_degree, rep_method.num_buckets) + 1)
        else:
            degree_counts = [0] * (graph.max_degree + 1)

        #For each node in k-hop neighbors, count its degree
        for kn in kneighbors:
            weight = 1 #unweighted graphs supported here
            degree = graph.node_degrees[kn]
            if rep_method.num_buckets is not None:
                try:
                    degree_counts[int(math.log(degree, rep_method.num_buckets))] += weight
                except:
                    print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
            else:
                degree_counts[degree] += weight
        return degree_counts

    #Get structural features for nodes in a graph based on degree sequences of neighbors
    #Input: graph, RepMethod
    #Output: nxD feature matrix
    def get_features(self, graph, rep_method, verbose = True):
        before_khop = time.time()
        #Get k-hop neighbors of all nodes
        khop_neighbors_nobfs = self.get_khop_neighbors(graph, rep_method)

        graph.khop_neighbors = khop_neighbors_nobfs

        if verbose:
            print("max degree: {}".format(graph.max_degree))
            after_khop = time.time()
            print("got k hop neighbors in time: {}".format(after_khop - before_khop))

        G_adj = graph.G_adj
        num_nodes = G_adj.shape[0]
        if rep_method.num_buckets is None: #1 bin for every possible degree value
            num_features = graph.max_degree + 1 #count from 0 to max degree...could change if bucketizing degree sequences
        else: #logarithmic binning with num_buckets as the base of logarithm (default: base 2)
            num_features = int(math.log(graph.max_degree, rep_method.num_buckets)) + 1
        feature_matrix = np.zeros((num_nodes, num_features))

        before_degseqs = time.time()
        for n in range(num_nodes):
            for layer in graph.khop_neighbors[n].keys(): #construct feature matrix one layer at a time
                if len(graph.khop_neighbors[n][layer]) > 0:
                    #degree sequence of node n at layer "layer"
                    deg_seq = self.get_degree_sequence(graph, rep_method, graph.khop_neighbors[n][layer], n)
                    #add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                    feature_matrix[n] += [(rep_method.alpha**layer) * x for x in deg_seq]
        after_degseqs = time.time()

        if verbose:
            print("got degree sequences in time: {}".format(after_degseqs - before_degseqs))

        return feature_matrix

    #Input: two vectors of the same length
    #Optional: tuple of (same length) vectors of node attributes for corresponding nodes
    #Output: number between 0 and 1 representing their similarity
    def compute_similarity(self, graph, rep_method, vec1, vec2, node_attributes = None, node_indices = None):
        dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2) #compare distances between structural identities
        if graph.node_attributes is not None:
            #distance is number of disagreeing attributes
            attr_dist = np.sum(graph.node_attributes[node_indices[0]] != graph.node_attributes[node_indices[1]])
            dist += rep_method.gammaattr * attr_dist
        return np.exp(-dist) #convert distances (weighted by coefficients on structure and attributes) to similarities

    #Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
    #Input: graph (just need graph size here), RepMethod (just need dimensionality here)
    #Output: np array of node IDs
    def get_sample_nodes(self, graph, rep_method, verbose = True):
        #Sample uniformly at random
        sample = np.random.permutation(np.arange(graph.N))[:rep_method.p]
        return sample

    #Get dimensionality of learned representations
    #Related to rank of similarity matrix approximations
    #Input: graph, RepMethod
    #Output: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
    def get_feature_dimensionality(self, graph, rep_method, verbose = True):
        p = int(rep_method.k*math.log(graph.N, 2)) #k*log(n) -- user can set k, default 10
        if verbose:
            print("feature dimensionality is {}".format(min(p, graph.N)))
        rep_method.p = min(p,graph.N)  #don't return larger dimensionality than # of nodes
        return rep_method.p

    #xNetMF pipeline
    def get_representations(self, graph, rep_method, verbose = True):
        #Node identity extraction
        feature_matrix = self.get_features(graph, rep_method, verbose)

        #Efficient similarity-based representation
        #Get landmark nodes
        if rep_method.p is None:
            rep_method.p = self.get_feature_dimensionality(graph, rep_method, verbose = verbose) #k*log(n), where k = 10
        elif rep_method.p > graph.N:
            print("Warning: dimensionality greater than number of nodes. Reducing to n")
            rep_method.p = graph.N
        landmarks = self.get_sample_nodes(graph, rep_method, verbose = verbose)

        #Explicitly compute similarities of all nodes to these landmarks
        before_computesim = time.time()
        C = np.zeros((graph.N,rep_method.p))
        for node_index in range(graph.N): #for each of N nodes
            for landmark_index in range(rep_method.p): #for each of p landmarks
                #select the p-th landmark
                C[node_index,landmark_index] = self.compute_similarity(graph,
                                                                       rep_method,
                                                                       feature_matrix[node_index],
                                                                       feature_matrix[landmarks[landmark_index]],
                                                                       graph.node_attributes,
                                                                       (node_index, landmarks[landmark_index]))

        before_computerep = time.time()

        #Compute Nystrom-based node embeddings
        W_pinv = np.linalg.pinv(C[landmarks])
        U,X,V = np.linalg.svd(W_pinv)
        Wfac = np.dot(U, np.diag(np.sqrt(X)))
        reprsn = np.dot(C, Wfac)
        after_computerep = time.time()
        if verbose:
            print("computed representation in time: {}".format(after_computerep - before_computerep))

        #Post-processing step to normalize embeddings (true by default, for use with REGAL)
        if rep_method.normalize:
            reprsn = reprsn / np.linalg.norm(reprsn, axis = 1).reshape((reprsn.shape[0],1))
        return reprsn
