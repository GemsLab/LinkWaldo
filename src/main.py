from netmf_emb import NetMF
from aa_emb import AA
from bine_emb import BiNE
from lapm_selector import LaPMSelector
from dg_selector import DGSelector
from mg_selector import MGSelector
from dg_bailout_selector import DGBailoutSelector
from mg_bailout_selector import MGBailoutSelector
from cn_selector import CNSelector
from js_selector import JSSelector
from aa_selector import AASelector
from sg_selector import SGSelector
from cg_selector import CGSelector
from bagging_ensemble import BaggingEnsemble
import networkx as nx
import cProfile
import argparse
import os

def parse_args():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--graph', '-G', type=str, required=True, help="The name of the graph to use.")
    parser.add_argument('--profile', '-p', type=str2bool, default=False, required=False, help="If True, then use cProfile.")
    parser.add_argument('--method', '-m', type=str, required=False, default='LinkWaldo', help="The method to use to select pairs.")
    parser.add_argument('--embedding_method', '-em', type=str, default='netmf2', required=False, help="The embedding method to use.")
    parser.add_argument('--k', '-k', type=int, default=10000000, required=False, help="How many pairs to return.")
    parser.add_argument('--force_emb', '-force_emb', type=str2bool, default=False, required=False, help="Whether to re-compute embedding if the file already exists.")
    parser.add_argument('--sampling_method', '-sm', type=str, default='static', required=False, help="static or temporal.")
    parser.add_argument('--percent_test', '-test', type=float, default=20.0, required=False, help="Percent of edges to use for testing, separated by '-'.")
    parser.add_argument('--seed', '-s', type=str, default=0, required=False, help="The random seed to run on.")
    parser.add_argument('--bipartite', '-bip', type=str2bool, default=False, required=False, help="True if graph is bipartite.")
    parser.add_argument('--exact_search_tolerance', '-tol', type=int, default=25000000, required=False, help="The level at which to search cells exactly vs. with LSH.")
    parser.add_argument('--output_override', '-oo', type=str, default=None, required=False, help="If specified, this path will override the default output path.")
    parser.add_argument('--num_groups', '-num_groups', type=int, default=None, required=False, help="How many groups to use.")
    parser.add_argument('--num_groups_alt', '-num_groups_alt', type=int, default=None, required=False, help="How many groups to use for 2nd/3rd grouping.")
    parser.add_argument('--DG', '-DG', type=str2bool, default=True, required=False, help="Whether to use grouping DG.")
    parser.add_argument('--SG', '-SG', type=str2bool, default=True, required=False, help="Whether to use grouping SG.")
    parser.add_argument('--CG', '-CG', type=str2bool, default=True, required=False, help="Whether to use grouping CG.")
    parser.add_argument('--bailout_tol', '-b_tol', type=float, default=0.25, required=False, help="The threshold for bailing out of a cell if the embeddings are uniformitive.")
    parser.add_argument('--bag_epsilon', '-bag_epsilon', type=float, default=1.0, required=False, help="The epsilon for the NMF+BAG baseline.")
    parser.add_argument('--skip_output', '-skip_output', type=str2bool, default=False, required=False, help="If True, don't write the found pairs.")
    return parser.parse_args()

def main(args, jupyter=False):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    seed = args.seed
    edgelist = os.path.join(ROOT_DIR, '../data/{}/train/{}_{}_seed_{}.txt'.format(args.sampling_method, args.graph, args.percent_test, seed))
    if not args.bipartite:
        G = nx.read_edgelist(edgelist)
    else:
        bip_edges = list()
        A = set()
        B = set()
        for line in open(edgelist, 'r'):
            a, b = line.strip().split()
            A.add(a)
            B.add(b)
            bip_edges.append((a, b))
        G = nx.Graph()
        G.add_nodes_from(A, bipartite=0)
        G.add_nodes_from(B, bipartite=1)
        G.add_edges_from(bip_edges)

    test_path = os.path.join(ROOT_DIR, '../data/{}/test/{}_{}_seed_{}.txt'.format(args.sampling_method, args.graph, args.percent_test, seed))

    output_dir = os.path.join(ROOT_DIR, '../output/{}/'.format(args.sampling_method))
    emb_path = os.path.join(output_dir, '{}_{}_{}_seed_{}.emb'.format(args.embedding_method, args.graph, args.percent_test, seed))
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    if not args.output_override:
        output_path = os.path.join(output_dir, '{}_{}_{}_{}_{}_{}_{}_k_{}.txt'.format(args.method, args.graph, args.embedding_method, args.percent_test, args.exact_search_tolerance, args.bailout_tol, seed,  args.k))
    else:
        output_path = args.output_override

    if args.embedding_method == 'netmf1':
        embeddings = NetMF(args.embedding_method, edgelist, test_path, emb_path, G, normalize=True, window_size=1)
    elif args.embedding_method == 'netmf2':
        embeddings = NetMF(args.embedding_method, edgelist, test_path, emb_path, G, normalize=True, window_size=2)
    elif args.embedding_method == 'bine':
        embeddings = BiNE(args.embedding_method, edgelist, test_path, emb_path, G, normalize=True)
    elif args.embedding_method == 'aa':
        embeddings = AA(args.embedding_method, edgelist, test_path, emb_path, G, normalize=True)
    if args.force_emb or not os.path.exists(emb_path):
        if os.path.exists(emb_path.replace('.emb', '_nodeX.npy')):
            os.remove(emb_path.replace('.emb', '_nodeX.npy'))
        embeddings.run(G)

    if args.method in {'lapm'}:
        sel = LaPMSelector(args.method, G,  args.k, embeddings, output_path, seed=seed, bipartite=args.bipartite)
        load_embeddings = True
    elif args.method in {'cn'}:
        sel = CNSelector(args.method, G,  args.k, embeddings, output_path, seed=seed, bipartite=args.bipartite)
        load_embeddings = False
    elif args.method in {'js'}:
        sel = JSSelector(args.method, G,  args.k, embeddings, output_path, seed=seed, bipartite=args.bipartite)
        load_embeddings = False
    elif args.method in {'aa'}:
        sel = AASelector(args.method, G,  args.k, embeddings, output_path, seed=seed, bipartite=args.bipartite)
        load_embeddings = False
    elif args.method in {'nmf+bag'}:
        sel = BaggingEnsemble(args.method, G,  args.k, embeddings, output_path, seed=seed, bipartite=args.bipartite)
        load_embeddings = False
    elif args.method == 'LinkWaldo':
        num_groupings = 0
        if args.DG:
            num_groupings += 1
        if args.SG:
            num_groupings += 1
        if args.CG:
            num_groupings += 1

        if num_groupings > 1:
            if args.bailout_tol > 0.0:
                sel = MGBailoutSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
            else:
                sel = MGSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
        else:
            if args.DG and args.bailout_tol > 0.0:
                sel = DGBailoutSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
            elif args.DG:
                sel = DGSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
            elif args.SG:
                sel = SGSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
            elif args.CG:
                sel = CGSelector(args.method, G,  args.k, embeddings, output_path, DG=args.DG, SG=args.SG, CG=args.CG, exact_search_tolerance=args.exact_search_tolerance, seed=seed, bipartite=args.bipartite)
        load_embeddings = True

    sel.num_groups = args.num_groups
    sel.num_groups_alt = args.num_groups_alt
    sel.bailout_tol = args.bailout_tol
    sel.bag_epsilon = args.bag_epsilon
    sel.skip_output = args.skip_output
            
    embeddings.load_data(load_embeddings=load_embeddings)

    if jupyter:
        return sel
            
    _time = sel.select()

    sel.write_res(_time)
    if not args.skip_output:
        sel.write()

if __name__ == "__main__":
    args = parse_args()
    if not args.profile:
        main(args)
    else:
        cProfile.run('main(args)')
