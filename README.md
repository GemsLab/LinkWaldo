# LinkWaldo

Caleb Belth, Alican Büyükçakır, and Danai Koutra. _A Hidden Challenge of Link Prediction: Which Pairs to Check?_. IEEE International Conference on Data Mining (ICDM), November 2020. [[Link to the paper](https://quickshift.xyz/public/assets/documents/belth-2020-ICDM-LinkWaldo.pdf)]

If used, please cite:
```bibtex
@inproceedings{belth2020hidden,
  title={A Hidden Challenge of Link Prediction: Which Pairs to Check?},
  author={Belth, Caleb and B{\"u}y{\"u}k{\c{c}}ak{\i}r, Alican and Koutra, Danai},
  booktitle={2020 IEEE International Conference on Data Mining (ICDM)},
  pages={831--840},
  year={2020},
  organization={IEEE}
}
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

`data/static/` contains the static graphs while `data/temporal/` contains the temporal graphs (Table II).

Subdirectories `test/` and `train/` within each contain the graphs after removing test edges (see "Generating train/test splits secton" above).

## Example usage (from `src/` dir)

Run on Yeast dataset with MG grouping:

`python main.py -G yeast -sm static -em netmf1 -k 10000`

Run on Yeast dataset with DG grouping:

`python main.py -G yeast -sm static -em netmf1 -k 10000 -SG False -CG False --bailout_tol 0.0`

#### NMF+BAG baseline

`python main.py -G yeast -m nmf+bag -k 10000 -bag_epsilon 1.0`

### Arguments


`--graph / -G {graph_name} (Required)` The graph on which to run.

`--k / -k {budget} (Optional; Default = 10M)` The budget (how many pairs to output). 

`--method / -m {method} (Optional; Default = 'LinkWaldo')` Specifies the method (one of `'LinkWaldo', 'lapm', 'cn', 'js', 'aa', 'nmf+bag'`).

`--embedding_method {the proximity method to use} (Optional; Default = 'netmf2')` Specifies the proximity method to use within equivalence classes (one of `'netmf1', 'netmf2', 'bine', 'aa'`).

`--DG / -DG {True/False} (Optional; Default = True)` Whether or not to use Degree Grouping (Sec. IV.A).

`--SG / -SG {True/False} (Optional; Default = True)` Whether or not to use Structural Grouping (Sec. IV.A).

`--CG / -CG {True/False} (Optional; Default = True)` Whether or not to use Community Grouping (Sec. IV.A).

`--exact_search_tolerance / -tol (Optional; Default = 25M)` The tolerance for exactly vs. approximately searching equivlance classes (Sec. IV.C and V.D).

`--bailout_tol / -b_tol (Optional; Default = 0.5)` (Sec. IV.C and V.D)

`--sampling_method / -sm {'static'/'temporal'} (Optional; Default = 'static')` Whether the graph is temporal or static (Table II).

`--num_groups / -num_groups {int} (Optional; Default = 25)` The number of groups to use for a grouping (Sec. V.D).

`--num_groups_alt / -num_groups_alt {int} (Optional; Default = 5)` If there are multiple groupings, this allows non-DG groupings to use a different number of groups (Sec. V.D).

`--percent_test / -test {float} (Optional; Default = 20.0)` The percentage of the graph used for evalution (see "Generating train/test splits secton" above).

`--seed / -s {int} (Optional; Default = 0)` The seed to run on (see "Generating train/test splits secton" above).

`--bipartite / -bip {True/False} (Optional; Default = False)` Whether or not the graph is bipartite.

`--output_override / -oo {None or file_path} (Optional; Default = None)` If None, the output will go to its default location (`data/{sample_method}/`), otherwise it will go to the path specified by this parameter.

`--bag_epsilon / -bag_epsilon {float} (Optional; Default = 1.0)` The epsilon parameter for the NMF+Bag baseline (Sec. V.B).

`--skip_output / -skip_output {True/False} (Optional; Default = False)` If True, not output files will be created.

`--force_emb / force_emb {True / False} (Optional; Default = False)` If True, then even if the embeddings have already been generated for the input graph, they will be re-generated.

