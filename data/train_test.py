import random
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
    parser.add_argument('--method', '-m', type=str, default='static', required=False, help="static or temporal.")
    parser.add_argument('--percent_test', '-test', type=float, default=20, required=False, help="Percent of edges to use for testing.")
    parser.add_argument('--seed', '-s', type=str, default=0, required=False, help="The random seed to start with.")
    return parser.parse_args()

def write_train_test(args, train_edges, test_edges):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    train_path = os.path.join(ROOT_DIR, '{}/train/{}_{}_seed_{}.txt'.format(args.method, args.graph, args.percent_test, args.seed))
    test_path = os.path.join(ROOT_DIR, '{}/test/{}_{}_seed_{}.txt'.format(args.method, args.graph, args.percent_test, args.seed))
    print('Train path: {}'.format(train_path))
    print('Test path: {}'.format(test_path))
    with open(train_path, 'w') as f:
        for u, v, *rest in train_edges:
            f.write('{} {}\n'.format(u, v))
    with open(test_path, 'w') as f:
        for u, v, label in test_edges:
            f.write('{} {} {}\n'.format(u, v, label))

def train_test_static(args, edgelist):
    '''
    Generate train and test splits for a static network.
    '''
    edges = list()
    for line in open(edgelist, 'r'):
        if line.startswith('#'):
            continue
        u, v = sorted(tuple(line.strip().split())[:2])
        edges.append((u, v))
    num_test = int(round(len(edges) * (args.percent_test / 100)))
    num_train = len(edges) - num_test

    test_edges = random.sample(edges, k=num_test)
    train_edges = sorted(set(edges).difference(test_edges))

    test_edges = list((edge[0], edge[1], 1) for edge in test_edges)
    return train_edges, test_edges

def train_test_temporal(args, edgelist):
    '''
    Generate train and test splits for a temporal network.
    '''
    edges = list()
    seen = set()
    for line in open(edgelist, 'r'):
        if line.startswith('#'):
            continue
        line = line.strip().split()
        u, v = line[:2]
        if (u, v) in seen or (v, u) in seen:
            continue
        seen.add((u, v))
        seen.add((v, u))
        edges.append(line)
    num_test = int(round(len(edges) * (args.percent_test / 100)))
    num_train = len(edges) - num_test
    train_edges = edges[:num_train]
    test_edges = edges[num_train:]
    assert(len(train_edges) == num_train and len(test_edges) == num_test)
    assert(num_train + num_test == len(edges))

    test_edges = list((edge[0], edge[1], 1) for edge in test_edges)
    return train_edges, test_edges

def main(args):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    edgelist = os.path.join(ROOT_DIR, '{}/{}.txt'.format(args.method, args.graph))

    random.seed(args.seed)
    if args.method == 'static':
        train_edges, test_edges = train_test_static(args, edgelist)
    elif args.method == 'temporal':
        train_edges, test_edges = train_test_temporal(args, edgelist)
    print('Train edges: {} Test edges: {}'.format(len(train_edges), len(test_edges)))
    write_train_test(args, train_edges, test_edges)

if __name__ == "__main__":
    args = parse_args()
    main(args)