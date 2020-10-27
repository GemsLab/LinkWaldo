from collections import defaultdict
import argparse
import sys
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
    parser.add_argument('--seed', '-s', type=str, default=0, required=False, help="The random seed to run on.")
    return parser.parse_args()

def get_sampling_method(graph):
    return 'static' if graph in {'yeast', 'dblp_citation', 'facebook_ego', 'protein', 'arxiv', 'epinions', 'protein_3847'} else 'temporal'

def get_k(graph):
    if graph == 'yeast':
        return 10000
    elif graph in {'dblp_citation', 'facebook_ego', 'movielens', 'protein', 'arxiv'}:
        return 100000
    elif graph in {'math_overflow', 'reddit', 'enron', 'epinions', 'facebook_temporal'}:
        return 1000000
    elif graph in {'digg', 'protein_3847'}:
        return 10000000

def bipartite(args):
    oo = '../output/experiments/main/{}_{}_{}_{}_{}'.format(args.graph, get_sampling_method(args.graph), get_k(args.graph), args.seed, 'bine')
    print(oo)
    _oo = '{}_deg.txt'.format(oo)
    if os.path.exists(_oo.replace('.txt', '_R.txt')):
        print('Skipping {}'.format(_oo))
    else:
        command = 'python main.py -G {} -sm {} -m LinkWaldo -em bine -k {} -SG False -CG False --seed {} --bailout_tol 0.0 -oo {} -bip True'.format(args.graph, get_sampling_method(args.graph), get_k(args.graph), args.seed, _oo)
        os.system(command)
    _oo = '{}_multi.txt'.format(oo)
    if os.path.exists(_oo.replace('.txt', '_R.txt')):
        print('Skipping {}'.format(_oo))
    else:
        command = 'python main.py -G {} -sm {} -m LinkWaldo -em bine -k {} --seed {} --bailout_tol 0.0 -oo {} -bip True'.format(args.graph, get_sampling_method(args.graph), get_k(args.graph), args.seed, _oo)
        os.system(command)

def main(args):
    if args.graph == 'movielens':
        bipartite(args)
        return
    for em in ['netmf1', 'netmf2', 'aa']:
        oo = '../output/experiments/main/{}_{}_{}_{}_{}'.format(args.graph, get_sampling_method(args.graph), get_k(args.graph), args.seed, em)
        print(oo)
        _oo = '{}_deg.txt'.format(oo)
        if os.path.exists(_oo.replace('.txt', '_R.txt')):
            print('Skipping {}'.format(_oo))
        else:
            command = 'python main.py -G {} -sm {} -m LinkWaldo -em {} -k {} -SG False -CG False --seed {} --bailout_tol 0.5 -oo {} --skip_output True'.format(args.graph, get_sampling_method(args.graph), em, get_k(args.graph), args.seed, _oo)
            os.system(command)
        _oo = '{}_multi.txt'.format(oo)
        if os.path.exists(_oo.replace('.txt', '_R.txt')):
            print('Skipping {}'.format(_oo))
        else:
            command = 'python main.py -G {} -sm {} -m LinkWaldo -em {} -k {} --seed {} --bailout_tol 0.5 -oo {} --skip_output True'.format(args.graph, get_sampling_method(args.graph), em, get_k(args.graph), args.seed, _oo)
            os.system(command)

if __name__ == "__main__":
    args = parse_args()
    main(args)