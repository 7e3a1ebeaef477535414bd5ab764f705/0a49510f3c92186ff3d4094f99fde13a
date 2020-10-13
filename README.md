# Online Learning of Graph Neural Networks: When can data be permanently deleted?


**This repository holds the anonymized code for the review process at ICLR 2021**


## Installation

1. Install [pytorch](https://pytorch.org/get-started/locally/) as suited to your
   OS / python package manager / CUDA version
1. Install [dgl](https://www.dgl.ai/pages/start.html) as suited to your
   OS / python package manager / CUDA version
1. Install [torch-geometric](https://github.com/rusty1s/pytorch_geometric)
1. Install other requirements via `pip install -r requirements.txt` within your
   copy of this repository. This will include mainly `pandas`, `seaborn`, and `scikit-learn`.

## Prepare datasets

Unzip data.zip such that the directory structure looks like this:

- `data/dblp-easy/`
- `data/dblp-hard/`
- `data/pharmabio/`

where `data` is located within your local copy of this repository.

## Example call to run an experiment

The following exemplary command will run an experiment with a GraphSAGE model (1 hidden layer with 32 hidden units) on the `dblp-easy` dataset starting evaluation at year 2003 while using 200 annual epochs.

```
python3 run_experiment.py --seed 42 --model gs-mean --n_hidden 32 --start cold --lr "0.005" --history 3 --n_layers 1 --weight_decay 0 --dropout 0.5 --rescale_lr 1.0 --initial_epochs 0 --annual_epochs 200 --dataset "dblp-easy" --t_start 2003  --save "results.csv"                       
```

The results.csv file can be reused for multiple runs (e.g. with different seeds, different models, different datasets), the script appends new results to the file.
Consult `python3 run_experiment.py -h` for more information.


## Visualize results

You can visualize with the `visualize.py` script:

```
python3 visualize.py --style "window size %RF" --hue model --col dataset --row start --nosharey --save plot.png results.csv
```

where `results.csv` is the results file. You can also provide multiple results files, which would then be concatenated before plotting.

## Aggregate results over time

Results can be aggregated to a tabular format via `tabularize.py`:

```
python3 tabularize.py -g model start annual_lr --save table.csv results.csv
```

where `results.csv` is the results file. You can also provide multiple results files, which would then be concatenated before plotting.
The option flag `-g` refers to columns on which the accuracy scores are grouped.
In this example, we would group by the `model` itself, the restart variant `start` (which could be either cold or warm), and the learning rate `annual_lr`. Thus, the results will get aggregated along all other columns including seeds and time steps.


## File Descriptions

| File                   | Description                                      |
| -                      | -                                                |
| analysis               | scripts to perform analyses                      |
| datasets.py            | dataset loading utilities                        |
| models                 | GNN implementations                              |
| README.md              | this file                                        |
| requirements.txt       | dependencies                                     |
| run_experiment.py      | main entry point for running a single experiment |
| tabularize.py          | aggregate results over time and output table                     |
| visualize.py           | visualize results                                |
