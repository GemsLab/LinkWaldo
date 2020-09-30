from emb import Emb
import networkx as nx
from BiNE.model.graph_utils import GraphUtils
import numpy as np
import random
import sys
from sklearn import preprocessing
import math
import os

class BiNE(Emb):
    def run(self, G):
        '''
        :G: a graph to run on.
        '''
        self.G = G

        print('Fitting BiNE.')

        def init_embedding_vectors(node_u, node_v, node_list_u, node_list_v, d=128):
            """
            initialize embedding vectors
            :param node_u:
            :param node_v:
            :param node_list_u:
            :param node_list_v:
            :return:
            """
            # user
            for i in node_u:
                vectors = np.random.random([1, d])
                help_vectors = np.random.random([1, d])
                node_list_u[i] = {}
                node_list_u[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
                node_list_u[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')
            # item
            for i in node_v:
                vectors = np.random.random([1, d])
                help_vectors = np.random.random([1, d])
                node_list_v[i] = {}
                node_list_v[i]['embedding_vectors'] = preprocessing.normalize(vectors, norm='l2')
                node_list_v[i]['context_vectors'] = preprocessing.normalize(help_vectors, norm='l2')



        def walk_generator(gul):
            """
            walk generator
            :param gul:
            :return:
            """
            gul.calculate_centrality('hits')
            p = 0.15
            large = 2
            maxT = 32
            minT = 1
            if large == 0:
                gul.homogeneous_graph_random_walks(percentage=p, maxT=maxT, minT=minT)
            elif large == 1:
                gul.homogeneous_graph_random_walks_for_large_bipartite_graph(percentage=p, maxT=maxT, minT=minT)
            elif large == 2:
                gul.homogeneous_graph_random_walks_for_large_bipartite_graph_without_generating(datafile=self.edgelist, percentage=p, maxT=maxT, minT=minT)
            return gul


        def get_context_and_negative_samples(gul):
            """
            get context and negative samples offline
            :param gul:
            :return: context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v
            """
            ws = 5
            large = 2
            ns = 4
            if large == 0:
                neg_dict_u, neg_dict_v = gul.get_negs(ns)
                print("negative samples is ok.....")
                context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.G_u, gul.walks_u, ws, ns, neg_dict_u)
                context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.G_v, gul.walks_v, ws, ns, neg_dict_v)
            else:
                neg_dict_u, neg_dict_v = gul.get_negs(ns)
                print("negative samples is ok.....")
                context_dict_u, neg_dict_u = gul.get_context_and_negatives(gul.node_u, gul.walks_u, ws, ns, neg_dict_u)
                context_dict_v, neg_dict_v = gul.get_context_and_negatives(gul.node_v, gul.walks_v, ws, ns, neg_dict_v)

            return context_dict_u, neg_dict_u, context_dict_v, neg_dict_v,gul.node_u,gul.node_v


        def skip_gram(center, contexts, negs, node_list, lam, pa):
            """
            skip-gram
            :param center:
            :param contexts:
            :param negs:
            :param node_list:
            :param lam:
            :param pa:
            :return:
            """
            loss = 0
            I_z = {center: 1}  # indication function
            for node in negs:
                I_z[node] = 0
            V = np.array(node_list[contexts]['embedding_vectors'])
            update = [[0] * V.size]
            for u in I_z.keys():
                if node_list.get(u) is  None:
                    pass
                Theta = np.array(node_list[u]['context_vectors'])
                X = float(V.dot(Theta.T))
                sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))
                update += pa * lam * (I_z[u] - sigmod) * Theta
                node_list[u]['context_vectors'] += pa * lam * (I_z[u] - sigmod) * V
                try:
                    loss += pa * (I_z[u] * math.log(sigmod) + (1 - I_z[u]) * math.log(1 - sigmod))
                except:
                    pass
                    # print "skip_gram:",
                    # print(V,Theta,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
            return update, loss


        def KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma):
            """
            KL-divergenceO1
            :param edge_dict_u:
            :param u:
            :param v:
            :param node_list_u:
            :param node_list_v:
            :param lam:
            :param gamma:
            :return:
            """
            loss = 0
            e_ij = edge_dict_u[u][v]

            update_u = 0
            update_v = 0
            U = np.array(node_list_u[u]['embedding_vectors'])
            V = np.array(node_list_v[v]['embedding_vectors'])
            X = float(U.dot(V.T))

            sigmod = 1.0 / (1 + (math.exp(-X * 1.0)))

            update_u += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * V
            update_v += gamma * lam * ((e_ij * (1 - sigmod)) * 1.0 / math.log(math.e, math.e)) * U

            try:
                loss += gamma * e_ij * math.log(sigmod)
            except:
                pass
                # print "KL:",
                # print(U,V,sigmod,X,math.exp(-X * 1.0),round(math.exp(-X * 1.0),10))
            return update_u, update_v, loss

            def top_N(test_u, test_v, test_rate, node_list_u, node_list_v, top_n):
                recommend_dict = {}
                for u in test_u:
                    recommend_dict[u] = {}
                    for v in test_v:
                        if node_list_u.get(u) is None:
                            pre = 0
                        else:
                            U = np.array(node_list_u[u]['embedding_vectors'])
                            if node_list_v.get(v) is None:
                                pre = 0
                            else:
                                V = np.array(node_list_v[v]['embedding_vectors'])
                                pre = U.dot(V.T)[0][0]
                        recommend_dict[u][v] = float(pre)

            precision_list = []
            recall_list = []
            ap_list = []
            ndcg_list = []
            rr_list = []

            for u in test_u:
                tmp_r = sorted(recommend_dict[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(recommend_dict[u]),top_n)]
                tmp_t = sorted(test_rate[u].items(), lambda x, y: cmp(x[1], y[1]), reverse=True)[0:min(len(test_rate[u]),top_n)]
                tmp_r_list = []
                tmp_t_list = []
                for (item, rate) in tmp_r:
                    tmp_r_list.append(item)

                for (item, rate) in tmp_t:
                    tmp_t_list.append(item)
                pre, rec = precision_and_recall(tmp_r_list,tmp_t_list)
                ap = AP(tmp_r_list,tmp_t_list)
                rr = RR(tmp_r_list,tmp_t_list)
                ndcg = nDCG(tmp_r_list,tmp_t_list)
                precision_list.append(pre)
                recall_list.append(rec)
                ap_list.append(ap)
                rr_list.append(rr)
                ndcg_list.append(ndcg)
            precison = sum(precision_list) / len(precision_list)
            recall = sum(recall_list) / len(recall_list)
            f1 = 2 * precison * recall / (precison + recall)
            map = sum(ap_list) / len(ap_list)
            mrr = sum(rr_list) / len(rr_list)
            mndcg = sum(ndcg_list) / len(ndcg_list)
            return f1, map, mrr, mndcg

        def nDCG(ranked_list, ground_truth):
            dcg = 0
            idcg = IDCG(len(ground_truth))
            for i in range(len(ranked_list)):
                id = ranked_list[i]
                if id not in ground_truth:
                    continue
                rank = i+1
                dcg += 1/ math.log(rank+1, 2)
            return dcg / idcg

        def IDCG(n):
            idcg = 0
            for i in range(n):
                idcg += 1 / math.log(i+2, 2)
            return idcg

        def AP(ranked_list, ground_truth):
            hits, sum_precs = 0, 0.0
            for i in range(len(ranked_list)):
                id = ranked_list[i]
                if id in ground_truth:
                    hits += 1
                    sum_precs += hits / (i+1.0)
            if hits > 0:
                return sum_precs / len(ground_truth)
            else:
                return 0.0

        def RR(ranked_list, ground_list):

            for i in range(len(ranked_list)):
                id = ranked_list[i]
                if id in ground_list:
                    return 1 / (i + 1.0)
            return 0

        def precision_and_recall(ranked_list,ground_list):
            hits = 0
            for i in range(len(ranked_list)):
                id = ranked_list[i]
                if id in ground_list:
                    hits += 1
            pre = hits/(1.0 * len(ranked_list))
            rec = hits/(1.0 * len(ground_list))
            return pre, rec

        def train_by_sampling(G, model_path):
            alpha, beta, gamma, lam = 0.01, 0.01, 0.1, 0.025
            max_iter = 100
            ws = 5
            print('======== experiment settings =========')
            print('alpha : %0.4f, beta : %0.4f, gamma : %0.4f, lam : %0.4f, ws : %d, max_iter : %d' % (alpha, beta, gamma, lam, ws, max_iter))
            print('========== processing data ===========')
            print("constructing graph....")
            gul = GraphUtils(model_path)
            gul.construct_training_graph(G)
            edge_dict_u = gul.edge_dict_u
            edge_list = gul.edge_list
            walk_generator(gul)
            print("getting context and negative samples....")
            context_dict_u, neg_dict_u, context_dict_v, neg_dict_v, node_u, node_v = get_context_and_negative_samples(gul)
            node_list_u, node_list_v = {}, {}
            init_embedding_vectors(node_u, node_v, node_list_u, node_list_v)
            last_loss, count, epsilon = 0, 0, 1e-3
        
            print("============== training ==============")
            for iter in range(0, max_iter):
                s1 = "\r[%s%s]%0.2f%%"%("*"* iter," "*(max_iter-iter),iter*100.0/(max_iter-1))
                loss = 0
                visited_u = dict(zip(node_list_u.keys(), [0] * len(node_list_u.keys())))
                visited_v = dict(zip(node_list_v.keys(), [0] * len(node_list_v.keys())))
                random.shuffle(edge_list)
                for i in range(len(edge_list)):
                    u, v, w = edge_list[i]
                    
                    length = len(context_dict_u[u])
                    random.shuffle(context_dict_u[u])
                    if visited_u.get(u) < length:
                        index_list = list(range(visited_u.get(u),min(visited_u.get(u)+1,length)))
                        for index in index_list:
                            context_u = context_dict_u[u][index]
                            neg_u = neg_dict_u[u][index]
                            # center,context,neg,node_list,eta
                            for z in context_u:
                                tmp_z, tmp_loss = skip_gram(u, z, neg_u, node_list_u, lam, alpha)
                                node_list_u[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_u[u] = index_list[-1]+3

                    length = len(context_dict_v[v])
                    random.shuffle(context_dict_v[v])
                    if visited_v.get(v) < length:
                        index_list = list(range(visited_v.get(v),min(visited_v.get(v)+1,length)))
                        for index in index_list:
                            context_v = context_dict_v[v][index]
                            neg_v = neg_dict_v[v][index]
                            # center,context,neg,node_list,eta
                            for z in context_v:
                                tmp_z, tmp_loss = skip_gram(v, z, neg_v, node_list_v, lam, beta)
                                node_list_v[z]['embedding_vectors'] += tmp_z
                                loss += tmp_loss
                        visited_v[v] = index_list[-1]+3

                    update_u, update_v, tmp_loss = KL_divergence(edge_dict_u, u, v, node_list_u, node_list_v, lam, gamma)
                    loss += tmp_loss
                    node_list_u[u]['embedding_vectors'] += update_u
                    node_list_v[v]['embedding_vectors'] += update_v

                delta_loss = abs(loss - last_loss)
                if last_loss > loss:
                    lam *= 1.05
                else:
                    lam *= 0.95
                last_loss = loss
                if delta_loss < epsilon:
                    break
                sys.stdout.write(s1)
                sys.stdout.flush()
            save_to_file(node_list_u,node_list_v,model_path)

        def ndarray_tostring(array):
            string = ""
            for item in array[0]:
                string += str(item).strip()+" "
            return string+"\n"

        def save_to_file(node_list_u, node_list_v, model_path):
            with open(model_path, 'w') as f:
                f.write('{} 128\n'.format(len(node_list_u) + len(node_list_v)))
                for u in node_list_u.keys():
                    if 'drug' in self.emb_path:
                        f.write('{} {}'.format(u.replace('u', ''), ndarray_tostring(node_list_u[u]['embedding_vectors'])))
                    else:
                        f.write('{} {}'.format(u, ndarray_tostring(node_list_u[u]['embedding_vectors'])))
                for v in node_list_v.keys():
                    if 'drug' in self.emb_path:
                        f.write('{} {}'.format(v.replace('m', ''), ndarray_tostring(node_list_v[v]['embedding_vectors'])))
                    else:
                        f.write('{} {}'.format(v, ndarray_tostring(node_list_v[v]['embedding_vectors'])))

        if 'drug' in self.emb_path:
            bip_edges = list()
            A = set()
            B = set()
            for a, b in G.edges():
                if b.startswith('D'):
                    b, a = a, b
                assert(not b.startswith('D'))
                a = 'u{}'.format(a)
                b = 'm{}'.format(b)
                A.add(a)
                B.add(b)
                bip_edges.append((a, b))
            G = nx.Graph()
            G.add_nodes_from(A, bipartite=0)
            G.add_nodes_from(B, bipartite=1)
            G.add_edges_from(bip_edges)
        train_by_sampling(G, self.emb_path)
        print('BiNE Fit.')
