# GraphLand Baselines

This repository provides the code for reproducing the results of graph foundation models and tabular baselines in the paper [GraphLand: Evaluating Graph Machine Learning Models on Diverse Industrial Data](https://arxiv.org/abs/2409.14500).

## Code

In `source` directory, one can find the source code for reproducing experiments in our paper. Each subdirectory is adapted from the corresponding open source repository:
- `gbdt` — [github.com/yandex-research/tabular-dl-tabr](https://github.com/yandex-research/tabular-dl-tabr)
- `anygraph` — [github.com/HKUDS/AnyGraph](https://github.com/HKUDS/AnyGraph)
- `opengraph` — [github.com/HKUDS/OpenGraph](https://github.com/HKUDS/OpenGraph)
- `tsgnn` — [github.com/benfinkelshtein/EquivarianceEverywhere](https://github.com/benfinkelshtein/EquivarianceEverywhere)
- `gcope` — [github.com/cshhzhao/GCOPE](https://github.com/cshhzhao/GCOPE)

## Environment

To install the enviroment suitable for all the considered baselines, do the following:

1. Install pixi according to https://pixi.sh/dev/installation/

2. Run `pixi install` to install the dependencies listed in `pixi.toml`

## Datasets

1. Download GraphLand datasets and unzip them into `datasets` directory

2. Run `pixi run python scripts/nfa.py` to prepare NFA features

3. Run `pixi run python scripts/convert.py` to convert datasets into the format required by each baseline


## Experiments

1. Change working directory to the specific baseline repository:
```
# for GBDT
cd source/gbdt

# for AnyGraph
cd source/anygraph/node_classification

# for OpenGraph
cd source/opengraph/node_classification

# for TS-GNN
cd source/tsgnn

# for GCOPE
cd source/gcope
```

2. Run the script:
```
pixi run python __run.py
```
