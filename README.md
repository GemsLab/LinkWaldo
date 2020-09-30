# LinkWaldo

***COMING SOON: Link to paper, citation, and more documentation!***

Caleb Belth, Alican Büyükçakır, and Danai Koutra. _A Hidden Challenge of Link Prediction: Which Pairs to Check?_. IEEE International Conference on Data Mining (ICDM), November 2020. [[Link to the paper](https://quickshift.xyz/public/assets/documents/belth-2020-ICDM-LinkWaldo.pdf)]

If used, please cite:
```bibtex
```

## Setup

```
$ git clone git@github.com:GemsLab/LinkWaldo.git
```

### Generating train/test splits

The `data/` directory contains a script `train_test.py` that can generate a train graph and test set for both static and temporal settings.

The training graph is created in the directory `data/static/train/` or `data/temporal/train` depending on the setting.

Similarly, the test set goes to the directory `data/static/test/` or `data/temporal/test`.

The script has already been run on the `yeast` dataset for five seeds.

#### Example usage:

`python train_test.py -G yeast -m static -test 20 --seed 0`

#### Arguments:

`--graph / -G {graph_name}` The name of the graph to use.

`--method / -m (static/temporal)` Whether the graph is static or temporal. Temporal graphs must have timestamps in the edgelist.
    
`--percent_test / -test [0, 100] (Optional; Default = 20)` Percent of edges to remove for testing. 20 was used in all experiments in the paper. 

`--seed / -s {seed}` The random seed. This allows multiple networks to be generated for evaluating variance in performance. For temporal networks, the edges removed will always be the same, but the file names will be different (cf. Sec. V.B).

### Requirements 

- `Python 3`
- `numpy`

## Data

#### Dataset Format

## Example usage (from `src/` dir)

`python main.py -G yeast -sm static -em netmf -k 10000`

`python main.py -G yeast -sm static -em netmf -k 10000 -SG False -CG False --bailout_tol 0.0`

### Arguments

