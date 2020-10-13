#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os

import numpy as np
import pandas as pd
import dgl
import torch_geometric as tg
import torch
import torch.nn.functional as F

# Models
from models import GraphSAGE
from models import GAT
from models import MLP
from models import MostFrequentClass
from models.geometric import SGNet
from models.graphsaint import train_saint, evaluate_saint
from models import geometric as geo

use_cuda = torch.cuda.is_available()

from datasets import load_data


def appendDFToCSV_void(df, csvFilePath, sep=","):
    """ Safe appending of a pandas df to csv file
    Source: https://stackoverflow.com/questions/17134942/pandas-dataframe-output-end-of-csv
    """
    if not os.path.isfile(csvFilePath):
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep)
    elif len(df.columns) != len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns):
        raise Exception(
            "Columns do not match!! Dataframe has " + str(len(df.columns)) + " columns. CSV file has " + str(
                len(pd.read_csv(csvFilePath, nrows=1, sep=sep).columns)) + " columns.")
    elif not (df.columns == pd.read_csv(csvFilePath, nrows=1, sep=sep).columns).all():
        raise Exception("Columns and column order of dataframe and csv file do not match!!")
    else:
        df.to_csv(csvFilePath, mode='a', index=False, sep=sep, header=False)


def compute_weights(ts, exponential_decay, initial_quantity=1.0, normalize=True):
    ts = torch.as_tensor(ts)
    delta_t = ts.max() - ts
    values = initial_quantity * torch.exp(- exponential_decay * delta_t)
    if normalize:
        # When normalizing, the initial_quantity is irrelevant
        values = values / values.sum()
    return values


def train(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None,
          backend='dgl'):
    model.train()
    reduction = 'none' if weights is not None else 'mean'
    for epoch in range(epochs):
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask], reduction=reduction)
        else:
            loss = F.cross_entropy(logits, labels, reduction=reduction)

        if weights is not None:
            loss = (loss * weights).sum()

        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Epoch {:d} | Loss: {:.4f}".format(epoch + 1, loss.detach().item()))


def evaluate(model, g, feats, labels, mask=None, compute_loss=True,
             backend='dgl'):
    model.eval()
    with torch.no_grad():
        inputs = (g, feats) if backend == 'dgl' else (feats, g)
        logits = model(*inputs)

        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        if isinstance(logits, np.ndarray):
            logits = torch.FloatTensor(logits)
        __max_vals, max_indices = torch.max(logits.detach(), 1)
        acc = (max_indices == labels).sum().float() / labels.size(0)

    return acc.item(), loss


def build_model(args, in_feats, n_hidden, n_classes, device, n_layers=1, backend='geometric',
                edge_index=None, num_nodes=None):
    if backend == 'geometric':
        if args.model == 'gs-mean':
            model = geo.GraphSAGE(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif args.model == "gcn":
            model = geo.GCN(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif args.model == "gat":
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            n_hidden_per_head = int(n_hidden / heads[0])
            model = geo.GAT(in_feats, n_hidden_per_head, n_classes, F.relu, args.dropout, 0.6, heads).to(device)
        elif args.model == "mlp":
            model = geo.MLP(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif args.model == "jknet":
            model = geo.JKNet(in_feats, n_hidden, n_classes, n_layers, F.relu, args.dropout).to(device)
        elif args.model == "sgnet":
            model = SGNet(in_channels=in_feats, out_channels=n_classes, K=n_layers).to(device)
        else:
            raise NotImplementedError
    else:
        if args.model == 'gs-mean':
            model = GraphSAGE(in_feats, n_hidden, n_classes,
                              n_layers, F.relu, args.dropout,
                              'mean').to(device)
        elif args.model == 'mlp':
            model = MLP(in_feats, n_hidden, n_classes,
                        n_layers, F.relu, args.dropout).to(device)
        elif args.model == 'mostfrequent':
            model = MostFrequentClass()
        elif args.model == 'gat':
            print("Warning, GAT doesn't respect n_layers")
            heads = [8, args.gat_out_heads]  # Fixed head config
            # Div num_hidden by heads for same capacity
            n_hidden_per_head = int(n_hidden / heads[0])
            assert n_hidden_per_head * heads[0] == n_hidden, f"{n_hidden} not divisible by {heads[0]}"
            model = GAT(1, in_feats, n_hidden_per_head, n_classes,
                        heads, F.elu, 0.6, 0.6, 0.2, False).to(device)
        else:
            raise NotImplementedError("Model not implemented")

    return model


def prepare_data_for_year(graph, features, labels, years, current_year, history, exclude_class=None,
                          device=None, backend='dgl'):
    print("Preparing data for year", current_year)
    # Prepare subgraph
    subg_nodes = torch.arange(features.size(0))[(years <= current_year) & (years >= (current_year - history))]

    subg_num_nodes = subg_nodes.size(0)

    if backend == 'dgl':
        subg = graph.subgraph(subg_nodes)
        subg.set_n_initializer(dgl.init.zero_initializer)
    elif backend == 'geometric':
        subg, __edge_attr = tg.utils.subgraph(subg_nodes,
                                              graph, relabel_nodes=True)
    else:
        raise ValueError("Unkown backend: " + backend)

    subg_features = features[subg_nodes]
    subg_labels = labels[subg_nodes]
    subg_years = years[subg_nodes]

    # Prepare masks wrt *subgraph*
    train_nid = torch.arange(subg_num_nodes)[subg_years < current_year]
    test_nid = torch.arange(subg_num_nodes)[subg_years == current_year]

    if exclude_class is not None:
        train_nid = train_nid[subg_labels[train_nid] != exclude_class]
        test_nid = test_nid[subg_labels[test_nid] != exclude_class]

    print("[{}] #Training: {}".format(current_year, train_nid.size(0)))
    print("[{}] #Test    : {}".format(current_year, test_nid.size(0)))
    if device is not None:
        if backend == 'geometric':
            subg = subg.to(device)
        subg_features = subg_features.to(device)
        subg_labels = subg_labels.to(device)
    return subg, subg_features, subg_labels, subg_years, train_nid, test_nid


RESULT_COLS = ['dataset',
               'seed',
               'model',
               'variant',
               'n_params',
               'n_hidden',
               'n_layers',
               'dropout',
               'history',
               'sampling',
               'batch_size',
               'cb_l',
               'cb_exp',
               'saint_coverage',
               'limited_pretraining',
               'initial_epochs',
               'initial_lr',
               'initial_wd',
               'annual_epochs',
               'annual_lr',
               'annual_wd',
               'start',
               'decay',
               'year',
               'epoch',
               'accuracy']


def main(args):
    if args.metric == "f1":
        RESULT_COLS[-1] = "f1-score"
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    has_parameters = args.model not in ['most_frequent']
    backend = 'geometric' if (args.model in ['jknet', 'sgnet'] or (args.sampling is not None and 'graphsaint' in args.sampling)) else 'dgl'
    print("Using backend:", backend)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if args.model == 'mostfrequent':
        #  Don't put things on GPU when using simple most frequent classifier
        device = torch.device("cpu")

    graph, features, labels, years = load_data(args.data_path, backend=backend)
    num_nodes = features.shape[0]
    num_edges = graph.number_of_edges() if backend == 'dgl' else graph.size(1)

    print("Min year:", years.min())
    print("Max year:", years.max())
    print("Number of nodes:", num_nodes)
    print("Number of edges:", num_edges)

    try:
        features = torch.FloatTensor(features.float())
    except AttributeError:
        features = torch.FloatTensor(features)

    labels = torch.LongTensor(labels)
    years = torch.LongTensor(years)
    n_classes = torch.unique(labels).size(0)

    features = features.to(device)
    labels = labels.to(device)
    print("Labels", labels.size())

    in_feats = features.shape[1]
    n_layers = args.n_layers
    n_hidden = args.n_hidden

    model = build_model(args, in_feats, n_hidden, n_classes, device,
                        n_layers=args.n_layers, backend=backend,
                        edge_index=graph, num_nodes=num_nodes)
    print(model)
    num_params = sum(np.product(p.size()) for p in model.parameters())
    print("#params:", num_params)
    if has_parameters:
        optimizer = torch.optim.Adam(model.parameters(),
                                    lr=args.lr,
                                    weight_decay=args.weight_decay)

    results_df = pd.DataFrame(columns=RESULT_COLS)

    def attach_score(df, year, epoch, accuracy):
        """ Partial """
        return df.append(
            pd.DataFrame(
                [[args.dataset,
                  args.seed,
                  args.model,
                  args.variant,
                  num_params,
                  args.n_hidden,
                  args.n_layers,
                  args.dropout,
                  args.history,
                  args.sampling,
                  args.batch_size,
                  args.cb_l,
                  args.cb_prob_exp,
                  args.saint_coverage,
                  args.limited_pretraining,
                  args.initial_epochs,
                  args.lr,
                  args.weight_decay,
                  args.annual_epochs,
                  args.lr * args.rescale_lr,
                  args.weight_decay * args.rescale_wd,
                  args.start,
                  args.decay,
                  year,
                  epoch,
                  accuracy]],
                columns=RESULT_COLS),
            ignore_index=True)

    known_classes = set()

    if not args.limited_pretraining and not args.start == 'cold' and args.initial_epochs > 0:
        # With 'limited pretraining' we run the initial epochs on
        # all data before cold start.
        # With cold start, no pretraining is needed.
        # When initial epochs are 0, no pretraining is needed either.
        # Exclusively the static model of experiment Q1 uses this pretraining
        # For other experiments, we set initial_epochs = 0.
        data = prepare_data_for_year(graph,
                                     features,
                                     labels,
                                     years,
                                     args.pretrain_until,
                                     10000,
                                     exclude_class=None,
                                     device=device,
                                     backend=backend)
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = data
        # Use all nodes of initial subgraph for training
        print("Using data until", args.pretrain_until, "for training")
        print("Selecting", subg_features.size(0), "of", features.size(0), "papers for initial training.")

        train_nids = torch.cat([train_nid, test_nid])  # use all nodes in subg for initial pre-training
        if args.model == 'mostfrequent':
            model.fit(None, subg_labels)
        elif args.model == "graphsaint":
            train_saint(model, optimizer, subg, subg_features, subg_labels,
                        mask=train_nid,
                        epochs=args.initial_epochs)
            acc, _ = evaluate_saint(model, subg, subg_features, subg_labels, mask=None,
                                    backend=backend, f1=args.metric == "f1")
            print(f"** Train Accuracy {acc:.4f} **")
        else:
            print("Subg labels", subg_labels.size())
            train(model, optimizer, subg, subg_features, subg_labels,
                  mask=train_nid,
                  epochs=args.initial_epochs, backend=backend)
            acc, _ = evaluate(model, subg, subg_features, subg_labels, mask=None,
                              backend=backend)
            print(f"** Train Accuracy {acc:.4f} **")

        known_classes |= set(subg_labels.cpu().numpy())
        print("Known classes:", known_classes)

    remaining_years = torch.unique(years[years > args.pretrain_until], sorted=True)

    for t, current_year in enumerate(remaining_years.numpy()):
        print(f"allocated: {torch.cuda.memory_allocated() / 1000000000} GB")
        torch.cuda.empty_cache()  # no memory leaks
        # Get the current subgraph

        if args.sampling is not None and ('graphsaint' in args.sampling):
            # GraphSAINT is trained on data until t - 1
            year_cutoff = current_year - 1
        else:
            # Other methods are trained on data until t (w/o test labels for t)
            year_cutoff = current_year

        data = prepare_data_for_year(graph,
                                     features,
                                     labels,
                                     years,
                                     year_cutoff,
                                     args.history,
                                     exclude_class=None,
                                     device=device,
                                     backend=backend)
        subg, subg_features, subg_labels, subg_years, train_nid, test_nid = data

        if args.decay is not None:
            # Use decay factor to weight the loss function based on time steps t
            weights = compute_weights(years[train_nid], args.decay, normalize=True).to(device)
        else:
            weights = None

        if args.history == 0:
            # No history means no uptraining at all!!!
            # Unused. For the static model (Exp. 1) we give a history frame but do no uptraining instead.
            epochs = 0
        elif args.limited_pretraining and t == 0:
            # Do the pretraining on the first history window
            # with `initial_epochs` instead of `annual_epochs`
            epochs = args.initial_epochs
        else:
            epochs = args.annual_epochs

        new_classes = set(subg_labels[train_nid].cpu().numpy()) - known_classes



        if args.start == 'legacy-cold':
            # Brute force re-init of model
            del model
            model = build_model(args, in_feats, n_hidden, n_classes, device, n_layers=args.n_layers,
                                edge_index=subg, num_nodes=subg_features.size(0))
        elif args.start == 'cold' or (args.start == 'hybrid' and new_classes):
            # NEW version, equivalent to legacy-cold, but more efficient
            model.reset_parameters()
        elif args.start == 'legacy-warm' or (args.start == 'hybrid' and not new_classes):
            # Legacy warm start: just keep old params as is
            # differs from new warm variant on unseen classes with cat. CE loss
            pass
        elif args.start == 'warm':
            if new_classes and has_parameters:
                print("Doing partial warm reinit")
                # If there are new classes:
                # 1) Save parameters of final layer
                # 2) Reinit parameters of final layer
                # 3) Copy saved parameters to new final layer
                known_class_ids = torch.LongTensor(list(known_classes))
                saved_params = [p.data.clone() for p in model.final_parameters()]
                model.reset_final_parameters()
                for i, params in enumerate(model.final_parameters()):
                    if params.dim() == 1:  # bias vector
                        params.data[known_class_ids] = saved_params[i][known_class_ids]
                    elif params.dim() == 2:  # weight matrix
                        params.data[known_class_ids, :] = saved_params[i][known_class_ids, :]
                    else:
                        NotImplementedError("Parameter dim > 2 ?")
        else:
            raise NotImplementedError("Unknown --start arg: '%s'" % args.start)

        known_classes |= new_classes
        print("Known classes:", known_classes)

        if has_parameters:
            # Build a fresh optimizer in both cases: warm or cold
            # Use rescaled lr and wd
            optimizer = torch.optim.Adam(model.parameters(),
                                         lr=args.lr * args.rescale_lr,
                                         weight_decay=args.weight_decay * args.rescale_wd)
        if args.model == 'mostfrequent':
            if epochs > 0:
                # Re-fit only if uptraining is in general allowed!
                model.fit(None, subg_labels[train_nid])

            acc, _ = evaluate(model,
                              subg,
                              subg_features,
                              subg_labels,
                              mask=test_nid,
                              compute_loss=False)
        elif args.sampling is not None and ("graphsaint" in args.sampling):
            if epochs > 0:
                print("SAINT")
                train_saint(model,
                            optimizer,
                            subg,
                            subg_features,
                            subg_labels,
                            sampling=args.sampling.replace('graphsaint_', ''),
                            mask=None,
                            epochs=epochs,
                            weights=weights,
                            walk_length=args.walk_length,
                            batch_size=args.batch_size,
                            coverage=args.saint_coverage,
                            l=args.cb_l,
                            prob_exp=args.cb_prob_exp)

            # Now, include time step t for GraphSAINT for evaluation.
            # GraphSAINT had only access to data until t - 1 before.
            subg, subg_features, subg_labels, subg_years, train_nid, test_nid = prepare_data_for_year(graph,
                                                                                                      features,
                                                                                                      labels,
                                                                                                      years,
                                                                                                      current_year,
                                                                                                      args.history,
                                                                                                      exclude_class=None,
                                                                                                      device=device,
                                                                                                      backend=backend)

            acc, _ = evaluate_saint(model,
                                    subg,
                                    subg_features,
                                    subg_labels,
                                    mask=test_nid,
                                    compute_loss=False,
                                    f1=args.metric == "f1")
        else:
            if epochs > 0:
                train(model,
                      optimizer,
                      subg,
                      subg_features,
                      subg_labels,
                      mask=train_nid,
                      epochs=epochs,
                      weights=weights,
                      backend=backend)

            acc, _ = evaluate(model,
                              subg,
                              subg_features,
                              subg_labels,
                              mask=test_nid,
                              compute_loss=False,
                              backend=backend)
        print(f"[{current_year} ~ Epoch {epochs}] Test Accuracy: {acc:.4f}")
        results_df = attach_score(results_df, current_year, epochs, acc)
        # input() # debug purposes
        # DROP ALL STUFF COMPUTED FOR CURRENT WINDOW (no memory leaks)
        del subg, subg_features, subg_labels, subg_years, train_nid, test_nid

    if args.save is not None:
        print("Saving final results to", args.save)
        appendDFToCSV_void(results_df, args.save)


DATASET_PATHS = {
    'dblp-easy': os.path.join('data', 'dblp-easy'),
    'dblp-hard': os.path.join('data', 'dblp-hard'),
    'pharmabio': os.path.join('data', 'pharmabio'),
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help="Specify model", default='gs-mean',
                        choices=['mlp', 'gs-mean', 'mostfrequent', 'gat', 'gcn', 'jknet', 'sgnet'])
    parser.add_argument('--sampling', type=str, choices=['graphsaint_rw', 'graphsaint_node', 'graphsaint_edge'],
                        default=None)
    parser.add_argument('--variant', type=str, default='',
                        help="Some comment on the model variant, useful to distinguish within results file")
    parser.add_argument('--dataset', type=str, help="Specify the dataset", choices=list(DATASET_PATHS.keys()),
                        default='pharmabio')
    parser.add_argument('--t_start', type=int,
                        help="The first evaluation time step. Default is 2004 for DBLP-{easy,hard} and 1999 for PharmaBio")

    parser.add_argument('--n_layers', type=int,
                        help="Number of layers/hops", default=2)
    parser.add_argument('--n_hidden', type=int,
                        help="Model dimension", default=64)
    parser.add_argument('--lr', type=float,
                        help="Learning rate", default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help="Weight decay", default=0.0)
    parser.add_argument('--dropout', type=float,
                        help="Dropout probability", default=0.5)

    parser.add_argument('--initial_epochs', type=int,
                        help="Train this many initial epochs", default=0)
    parser.add_argument('--annual_epochs', type=int,
                        help="Train this many epochs per year", default=200)
    parser.add_argument('--history', type=int,
                        help="How many years of data to keep in history", default=100)

    parser.add_argument('--gat_out_heads',
                        help="How many output heads to use for GATs", default=1, type=int)
    parser.add_argument('--rescale_lr', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--rescale_wd', type=float,
                        help="Rescale factor for learning rate and weight decay after pretraining", default=1.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_neighbors', type=int, default=1,
                        help="How many neighbors for control variate sampling")
    parser.add_argument('--limit', type=int, default=None,
                        help="Debug mode, limit number of papers to load")
    parser.add_argument('--batch_size', type=int, default=1000,
                        help="Number of seed nodes per batch for sampling")
    parser.add_argument('--test_batch_size', type=int, default=10000,
                        help="Test batch size (testing is done on cpu)")
    parser.add_argument('--num_workers', type=int, default=8, help="How many threads to use for sampling")
    parser.add_argument('--limited_pretraining', default=False, action="store_true",
                        help="Perform pretraining on the first history window.")
    parser.add_argument('--decay', default=None, type=float, help="Paramater for exponential decay loss smoothing")
    parser.add_argument('--save_intermediate', default=False, action="store_true",
                        help="Save intermediate results per year")
    parser.add_argument('--save', default=None, help="Save results to this file")
    parser.add_argument('--start', default='legacy-warm',
                        choices=['cold', 'warm', 'hybrid', 'legacy-cold', 'legacy-warm'],
                        help="Cold retrain from scratch or use warm start.")
    parser.add_argument("--mc_pool_size", default=1000, type=int)
    parser.add_argument("--mc_pool_smoothing", default=0.9, type=float)
    parser.add_argument("--mc_pool_alpha_mc", default=1., type=float)
    parser.add_argument("--mc_pool_alpha_o", default=1., type=float)
    parser.add_argument("--walk_length", default=2, type=int, help="Walk length for GraphSAINT random walk sampler")
    parser.add_argument("--saint_coverage", default=500, type=int)
    parser.add_argument("--cb_prob_exp", default=1, type=float)
    parser.add_argument("--cb_l", default=0.1, type=float)
    parser.add_argument("--cb_precompute", default=False, type=bool)
    parser.add_argument("--metric", default="acc", type=str, choices=["acc", "f1"])

    ARGS = parser.parse_args()
    if ARGS.save is None:
        print("**************************************************")
        print("*** Warning: results will not be saved         ***")
        print("*** consider providing '--save <RESULTS_FILE>' ***")
        print("**************************************************")

    # Handle dataset argument to get path to data
    try:
        ARGS.data_path = DATASET_PATHS[ARGS.dataset]
    except KeyError:
        print("Dataset key not found, trying to interprete as raw path")
        ARGS.data_path = ARGS.dataset
    print("Using dataset with path:", ARGS.data_path)

    # Handle t_start argument
    if ARGS.t_start is None:
        try:
            ARGS.t_start = {
                'dblp-easy': 2004,
                'dblp-hard': 2004,
                'pharmabio': 1999,
                'dblp-full': 2004
            }[ARGS.dataset]
            print("Using t_start =", ARGS.t_start)
        except KeyError:
            print("No default for dataset '{}'. Please provide '--t_start'.".format(ARGS.dataset))
            exit(1)

    # Backward compatibility:
    # current implementation actually uses 'pretrain_until'
    # as last timestep / year *BEFORE* t_start
    ARGS.pretrain_until = ARGS.t_start - 1

    main(ARGS)
