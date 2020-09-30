from selector import Selector
from scipy import linalg
from scipy.sparse import csr_matrix, csc_matrix, lil_matrix, coo_matrix
from scipy.sparse import csgraph
from scipy.sparse import random as randsparse
from time import time
import scipy.sparse as sparse
import networkx as nx
import numpy as np
import scipy as sp
import math
import random

class BaggingEnsemble(Selector):
    """
    Bagging ensemble that is described in the paper
    Duan et al. 2017 - An Ensemble Approach to Link Prediction.
    The method first samples subgraphs from the network, and then applies
    Symmetric NMF to get factorization that will be used in Link Prediction later.
    The link prediction scores of each component in the ensemble are iteratively combined
    to get the resulting predictions.
    The method has three variants:
        (1) Random Node Bagging (i.e. NMF(Node+) in the paper)
        (2) Edge Bagging (i.e. NMF(Edge+) in the paper)
        (3) Biased Edge Bagging (i.e. NMF(Biased+) in the paper)
    """

    def select(self, verbosity=10000):
        t0 = time()

        self.mu = 0.1    # expected number of appearances of each node in each component. Recommended number.
        self.f = 0.1      # fraction of nodes to be selected for ensemble. Recommended number.
        self.rho = 0.75   # fraction of edges to be selected for ensemble. Recommended number.

        self.n = self.G.number_of_nodes()

        self.ensemble_size = int(self.mu / self.f**2)     # Recommendation with theoretical grounding. Equal to 10 with these params.
        print('Ensemble size = {}'.format(self.ensemble_size))
        # self.ensemble_size = 2                  # for testing

        # ========================================================
        # ===== THESE BELOW SHOULD BE PARAMETERS OF THE MODEL ====
        # ========================================================
        self.k # number of links to be returned by top epsilon-k prediction. Authors use 1e-5, 1e-6

        # Attention: Runtime increases quadratically with approximation rank r.
        self.r = 50      # lower rank to be approximated by Symmetric NMF.

        # self.sampling_method = "random_node_bagging"
        # self.sampling_method = "edge_bagging"
        self.sampling_method = "biased_edge_bagging"

        # edge filter is default true. If false, it yields worse models..
        self.edge_filter = True

        # ======= END OF THE PARAMETERS ============================

        self.ensemble = None    # Ensemble that will hold the components.
        self.combined_preds = None

        # Initialize empty ensemble.
        self.ensemble = []

        # Generate ensemble components
        for i in range(self.ensemble_size):
            # Sample the nodeset according to the sample method chosen.
            if(self.sampling_method == "random_node_bagging"):
                nodeset = self._get_nodeset_random_node_bagging(self.G)
                G_sub = self.G.subgraph(nodeset)

                # Factorize W_sub and get F.
                # As a result, we get W_sub ~= F * F^T.
                # F = self._symmetric_NMF(G_sub)
                F, F_corresp = self._symmetric_NMF(G_sub)

                # Extract S and R matrices from F.
                # S is the reordered columns of F and R is the corresponding permutations.
                # Notice that S and R are trimmed so that only the essential
                # links existing. Not close to zero ones.
                S, R = self._generate_S_R(F, F_corresp)

                # Component predictions as a sparse matrix
                preds_matrix = self._get_link_prediction_matrix(S, R)
                # preds_matrix = self._remove_already_existing(preds_matrix)

                self.ensemble.append(preds_matrix)

            elif(self.sampling_method == "edge_bagging"):
                nodeset = self._get_nodeset_edge_bagging(self.G)
                # print("subset size: {}".format(len(nodeset)))

                # Generate the induced subgraph
                G_sub = self.G.subgraph(nodeset)

                # In the induced subgraph, apply Edge Filter with Pref. Attachment
                # Reasoning: Because low number of edge nodes will tend to make less links.
                if(self.edge_filter == True):
                    G_sub = self._apply_edge_filter(G_sub, self.rho)

                # W_sub = nx.to_scipy_sparse_matrix(G_sub, nodelist=self.G.nodes())
                # F = self._symmetric_NMF(W_sub)
                F, F_corresp = self._symmetric_NMF(G_sub)

                S, R = self._generate_S_R(F, F_corresp)
                preds_matrix = self._get_link_prediction_matrix(S, R)
                self.ensemble.append(preds_matrix)

            elif(self.sampling_method == "biased_edge_bagging"):
                nodeset = self._get_nodeset_biased_edge_bagging(self.G)
                # print("subset size: {}".format(len(nodeset)))

                # Generate the induced subgraph
                G_sub = self.G.subgraph(nodeset)

                # In the induced subgraph, apply Edge Filter with Pref. Attachment
                # Reasoning: Because low number of edge nodes will tend to make less links.
                if(self.edge_filter == True):
                    G_sub = self._apply_edge_filter(G_sub, self.rho)

                # W_sub = nx.to_scipy_sparse_matrix(G_sub, nodelist=self.G.nodes())
                # F = self._symmetric_NMF(W_sub)
                F, F_corresp = self._symmetric_NMF(G_sub)

                S, R = self._generate_S_R(F, F_corresp)
                preds_matrix = self._get_link_prediction_matrix(S, R)
                self.ensemble.append(preds_matrix)

            else:
                print("Unidentified sampling method.")
                return

        # Now, combine predictions of the ensemble, resulting a dict
        self.combined_preds = self._combine_ensemble_predictions()

        self.pairs = set()
        for pair, val in self.combined_preds.items():
            v1, v2 = pair
            v1, v2 = self.embeddings.node_names_to_nodes[v1], self.embeddings.node_names_to_nodes[v2]
            if v1.name <= v2.name:
                self.pairs.add((v1, v2))
            else:
                self.pairs.add((v2, v1))

        print(len(self.pairs))

        t1 = time()

        print('====== {} pairs selected in {} seconds, {} of them pos. ======'.format(len(self.pairs), t1 - t0, self.true_pos()))
        print('R = {} P = {} % = {}'.format(self.recall(), self.precision(), self.prune_percent()))

        return t1 - t0

    def _predict(self, u, v):
        '''
        :u: a node
        :v: a node

        :return: The score for edge (u, v)
        '''
        if(u > v):      # Predictions are only in upper triangular.
            u, v = v, u  # If the asked link is in the lower, triangular, swap indices.

        if(int(u) >= self.n or int(v) >= self.n):   # Returns 0 for the unseen nodes?
            return 0.0                              # TODO: take a look at this.
        return self.combined_preds[int(u), int(v)]

    def _get_nodeset_random_node_bagging(self, G):
        """
        G: a networkx graph.
        ---

        Generates a subset of nodes, as in Random Node Bagging.
        - Node Uptake rule is coded as default.
        Returns the ids of the subset of nodes in a list.
        """
        # Limit on the number of nodes in the subset.
        # Number can exceed this value while adding multiple nodes at once.
        subset_size_limit = int(self.G.number_of_nodes() * self.f)

        # Generate random node set as follows:
        # - Start from a random node. Add it and its neighbors (Node Uptake Rule).
        # - Repeat until the subset_size_limit is reached.
        subset_nodes = set()

        while(len(subset_nodes) < subset_size_limit):
            rd = random.choice(list(self.G.nodes()))
            subset_nodes.add(rd)         # Add the node

            # Node Uptake Rule for Node Bagging: Instead of picking all nodes at random,
            # pick the neighbors whenever you pick a random node.
            neighs = [n for n in self.G.neighbors(rd)]
            subset_nodes.update(neighs)   # Add its neighbors..

        # Return the induced subset as a list, to be used
        # to extract subgraph from it later.

        return list(subset_nodes)

    def _get_nodeset_edge_bagging(self, G):
        """
        G: a networkx graph.
        ----

        Generates a subset of nodes, as in Edge Bagging.
        - Node Uptake rule is coded as default.
        - Edge Filter using Preferential Attachment is coded as default.
        Returns the ids of the subset of nodes in a list.
        """
        # Limit on the number of nodes in the subset.
        # Number can exceed this value while adding multiple nodes at once.
        subset_size_limit = int(self.G.number_of_nodes() * self.f)

        # Generate random node set as follows:
        # - Start from a random node, Add it and a portion of its neighbors.
        # but this time, also add the neighbors of the picked neighbors. (Node Uptake)
        # - Jump to a neighbor. IF there is none, jump to a random node.
        # - Repeat until the subset_size_limit is reached.

        subset_nodes = set()

        # Add an initial node.
        rd = random.choice(list(self.G.nodes()))
        subset_nodes.add(rd)         # Add the node

        while(len(subset_nodes) < subset_size_limit):
            # Node Uptake Rule for Edge Bagging: When you pick a random node,
            # For every node, add its neighborhood to the subset.
            neighs = [n for n in self.G.neighbors(rd)]

            # Add all neighbors, but then jump to some neighbor.
            subset_nodes.update(neighs)   # Add its neighbors..

            if(len(neighs) == 0):
                # No neighbor. Jump to a random node.
                rd = random.choice(list(self.G.nodes()))
            else:
                # Jump to a random neighbor
                rd = random.choice(neighs)

            if(rd in subset_nodes):     # If we end up trying to add an existing node,
                rd = random.choice(list(self.G.nodes()))    # then looping. jump to somewhere else.

            subset_nodes.add(rd)

        # Return the induced subset as a list, to be used
        # to extract subgraph from it later.

        return list(subset_nodes)

    def _apply_edge_filter(self, G_sub, rho):
        """
        G_sub: a networkx graph, which is the induced subgraph after edge bagging.
        rho: sampling ratio. Only the top (m_sub * rho) edges will be left in the
        induced subgraph.
        ---

        Performs "Edge Filter" using Preferential Attachment, as described in Section (3.4)
        Sorts the edges (i, j) according to minimum length of each pair's neighborhood,
        and removes the ones with lower minimum lengths.
        """
        # Number of edges that will be left at the end of this filter process.
        print("node subset size before edge filter: {}".format(G_sub.number_of_edges()))
        filtered_num_edges = int( G_sub.number_of_edges() * rho )

        for u, v in G_sub.edges():
            # print(u, v)
            if( u > v ):                # Fix if it is not compatible to
                u, v = v, u             # our upper-triangular sparse matrix..

            # Get the length of neighbors of each vertex and pick minimum to sort later.
            ne1 = len(self.G[u])
            ne2 = len(self.G[v])
            min_num_neighs = min(ne1, ne2)

            # Add the minimum number found as an attribute to work later.
            G_sub[u][v]["min_num_neighs"] = min_num_neighs

        # Initialize the resulting subgraph with filtered edges.
        G_sub_filtered = nx.Graph()

        count = 0
        # Sort the edges acc. to min num of neighs, and pick only the ones
        # that are the top (m_sub * rho) ones.
        for u, v, data in sorted(G_sub.edges(data=True), key=lambda x: x[2]["min_num_neighs"], reverse=True):
            # print("{} {} {} ".format(u, v, data["min_num_neighs"]))
            G_sub_filtered.add_edge(u, v)
            count += 1
            if(count >= filtered_num_edges):
                break

        print("node subset size after edge filter: {}".format(G_sub_filtered.number_of_nodes()))

        return G_sub_filtered

    def _get_nodeset_biased_edge_bagging(self, G):
        """
        G: a networkx graph.
        ---

        Generates a subset of nodes, as in Biased Edge Bagging.
        The difference between Edge Bagging and Biased Edge Bagging is that
        when the node set Ns is grown, a random adjacent node is not selected.
        Rather, an adjacent node with the least number of times selected
        in the previous ensemble components.
        - Node Uptake rule is coded as default.
        - Edge Filter using Preferential Attachment is coded as default.
        Returns the ids of the subset of nodes in a list.
        """
        # Limit on the number of nodes in the subset.
        # Number can exceed this value while adding multiple nodes at once.
        subset_size_limit = int(self.G.number_of_nodes() * self.f)

        subset_nodes = set()

        # Add an initial node.
        rd = random.choice(list(self.G.nodes()))
        subset_nodes.add(rd)         # Add the node

        while(len(subset_nodes) < subset_size_limit):
            # Node Uptake Rule for Edge Bagging: When you pick a random node,
            # For every node, add its neighborhood to the subset.
            neighs = [n for n in self.G.neighbors(rd)]
            subset_nodes.update(neighs)   # Add its neighbors..

            if(len(neighs) == 0):
                # No neighbor. Jump to a random node.
                rd = random.choice(list(self.G.nodes()))
            else:
                # Jump to a random neighbor of the current one,
                # but this time: pick a one that wasn't picked in the ensemble before,
                # or one that is picked the least.
                mn, mval = -1, np.inf   # seen node id with its min appearance val
                for n in neighs:
                    if("nac" not in G[n]):  # node does not have appearance count value
                        pass                # just disregard it.
                    else:
                        if( self.G[n]["nac"] < mval):
                            mn = n                    # update the best minimum
                            mval = self.G[n]["nac"]

                # if mn is still not assigned, then no neighbors have
                # appearance counts. in that case, pick a random one.
                if(mn == -1):
                    mn = random.choice(neighs)
                    if(mn in subset_nodes): # already added node. jump to somewhere else.
                        mn = random.choice(list(self.G.nodes()))

                # pick mn for your next neighbor.
                rd = mn

            subset_nodes.add(rd)

        for n in subset_nodes:          # Increment appearance counts.
            if("nac" in G[n]):
                self.G.nodes[n]["nac"] += 1
            else:
                self.G.nodes[n]["nac"] = 1

        # Return the induced subset as a list, to be used
        # to extract subgraph from it later.
        return list(subset_nodes)

    def _symmetric_NMF(self, G, beta=0.5, num_iter=50):
        """
        G: a networkx graph.
        beta: parameter to weight each new step's importance in latent space.
        num_iter: number of iterations in NMF
        ---

        Performs Nonnegative Matrix Factorization to weight matrix W
        so that W ~= F F^T
        """
        W = nx.to_scipy_sparse_matrix(G, nodelist=G.nodes())

        n_sub = G.number_of_nodes()
        F_corresp = G.nodes()

        # Initiate F as a random matrix
        F = np.random.rand(n_sub, self.r)

        # Apply Symmetric NMF below.
        F_new = F
        err_new = np.inf

        print('Symmetric NMF starts...')

        for i in range(num_iter):
            err = err_new

            # Equation 3 at Duan et al. 2017 TKDE
            numer = W.dot(F)
            denom = F.dot(F.transpose().dot(F))
            # F_new = (1.0 - beta) * F + F * ( beta * (numer/denom) )
            F_new = F * ((1.0 - beta) * np.ones((n_sub, self.r))  + beta * (numer / denom))

            err_new = np.linalg.norm( W - F_new.dot(F_new.transpose()) )

            if(i % 5 == 0):
                print('Iteration {}. Frobenius norm: {}'.format(i, err_new))

            F = F_new

            if(err_new > err):      # This does not happen, just in case.
                break

        print('Symmetric NMF completed. Frobenius norm: {}'.format(err_new))
        F = F_new

        return F, F_corresp

    def _generate_S_R(self, F, F_corresp, ignore_threshold = 0):#1e-8):
        """
        F: a numpy matrix (dense) obtained as a result of NMF.
        ignore_threshold: a threshold to prune S and R matrices.
                          below this value, elements in S won't participate in
                          the calculation of link predictions (too small)
        ----

        Given F, generate two matrices S and R where
        S is the matrix F with all columns are sorted in descending order
        and
        R is the corresponding column permutation matrix for the S matrix.
        S and R are returned as lists, where only the nonzero values and
        corresponding indices exist in them.
        """

        sorted_cols = []        # S matrix as list -- will come in handy
        sorted_vals = []        # R matrix as list

        for j in range(self.r):
            # trim seq so that only the indices with matrix's nonzero values
            # exist and the rest are gone.
            # e.g. [0, 1, 2, 3, 4] ---> [0, 1, 2, 4] because the entry in the
            # current column's 3rd row was zero.

            vals = F[:, j]
            col_indices = F_corresp #[int(i) for i in F_corresp] #F_corresp

            sorted_val, sorted_col = zip(*sorted(zip(vals, col_indices), reverse=True))

            # Prune the lists so that they will have the values for only
            # numbers that are considerably larger than zero.
            # Added an ignore_threshold for this...

            # Find the first index with entry zero to disregard the rest (smaller)
            zero_idx = next((i for i, x in enumerate(sorted_val) if x == 0), None)

            # Add only the ones that come before zero entries. Remember: vals are sorted.
            sorted_cols.append(sorted_col[:zero_idx])
            sorted_vals.append(sorted_val[:zero_idx])

            # print(sorted_col)

        return sorted_vals, sorted_cols    # S, R

    def _get_link_prediction_matrix(self, sorted_vals, sorted_cols):
        """
        sorted_vals: S matrix in a list of columns format
        sorted_cols: R matrix in a list of columns format.
        epsilon: total error that is allowed in "Top epsilon-k" predictions. Recommended value is 1.0.
        ----

        Using S and R, and by traversing the loop as in Section (2.1) of the paper,
        link predictions are generated in the form of a dict.
        """
        epsilon = self.bag_epsilon
        print(epsilon)

        link_approx = dict()

        # According to the loop in Section 2.1, fill the new matrix
        # that will have the link prediction approximation.

        threshold = math.sqrt(epsilon / self.r)  # threshold with theor. guarantees
        threshold_sqrd = epsilon / self.r

        print("Theoretical threshold sqrt(eps / r): {}".format(threshold))

        for c in range(self.r):   # For each column c,
            for i in range(len(sorted_vals[c])):
                if(sorted_vals[c][i] >= threshold):     # outer loop, until f_p (as in paper)
                    for j in range(i+1, len(sorted_vals[c])):
                        link_pred_value = 0.0

                        # indices to be filled in the link prediction matrix
                        idx1, idx2 = sorted_cols[c][i], sorted_cols[c][j]
                        # print(idx1, idx2)

                        # if the element corresponding to the current indices
                        # already exist in the original matrix, then no need to
                        # calculate, since it will not be predicted as a link anyways.
                        if(self.G.has_edge(idx1, idx2)):
                            continue

                        if(sorted_vals[c][i] * sorted_vals[c][j] < threshold_sqrd):
                            break
                        else:
                            # Add the link value of the found connection,
                            # but only if it's not a self link.
                            if(idx1 != idx2):       # Make sure it's not a self link.
                                # Wasn't a self link. Will add this value to the
                                # correct index in the prediction matrix.
                                # Use only upper triangular
                                if(idx2 < idx1):            # SWAP indices if necessary
                                    idx1, idx2 = idx2, idx1

                                link_pred_value = sorted_vals[c][i] * sorted_vals[c][j]

                                # Either initialize or update the link pred val.
                                if((idx1, idx2) in link_approx):
                                    link_approx[idx1, idx2] += link_pred_value
                                else:
                                    link_approx[idx1, idx2] = link_pred_value

        return link_approx #lp_coo      # dict of link pred scores. (u, v, score)

    def _combine_ensemble_predictions(self):
        """
        Combine ensemble components' link predictions by
        iteratively taking union of the predictions and pruning them to
        top-k predictions.

        Returns a sparse matrix where entries are predictions for the link pred task.
        """
        # Start the combined prediction as the first component itself.
        union = self.ensemble[0]

        for i in range(len(self.ensemble)):
            # Take union of the combined prediction matrix and the next component
            union = self._union_sort_trim_two_sparse_matrices(union, self.ensemble[i])

        return union

    def _union_sort_trim_two_sparse_matrices(self, X, Y):
        """
        X: dict of link pred scores. (u, v, score)
        Y: dict of link pred scores. (u, v, score)
        ---

        Two dicts can have common values that differ in magnitude.
        When combining predictions, the higher of those values should be picked
        according to the proposed ensemble approach.
        This method
            (1) takes elementwise maximum of two dictionaries
        See: https://stackoverflow.com/questions/19311353/element-wise-maximum-of-two-sparse-matrices
        Also notice iterkeys() don't work in Python 3.
        Instead, keys() is used in the implementation.
            (2) sorts the resulting dict according to prediction values
            (3) prunes if the resulting dict has more than k elements,
                so that it can have at most k predictions in the end.

        Returns: dict that is the (normalized and ready) union of two component predictions.
        """
        keys = list(set(X.keys()).union(Y.keys()))

        XmaxY = [max(X.get((i, j), 0), Y.get((i, j), 0)) for i, j in keys]

        # Now, sort the elementwise max matrix (union matrix)
        u_list, v_list = zip(*keys)
        sorted_XmaxY = sorted(zip(u_list, v_list, XmaxY), key=lambda v: v[2], reverse=True)

        # Take only the first k elements according to their prediction value.
        if(len(u_list) > self.k):
            # Trim...
            sorted_XmaxY = sorted_XmaxY[:self.k]
            u_list, v_list, XmaxY = map(list, zip(*sorted_XmaxY))

        result = dict()         # resulting dictionary of trimmed scores.
        for i in range(len(u_list)):
            result[u_list[i], v_list[i]] = XmaxY[i]

        return result
