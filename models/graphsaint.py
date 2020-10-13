"""
GraphSAINT: Graph Sampling Based Inductive Learning Method
Paper: https://arxiv.org/abs/1907.04931 (ICLR 2020)
Author's Code: https://github.com/GraphSAINT/GraphSAINT
Employed Implementation: https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html
"""
import numpy as np
import torch as th
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch_geometric.data import GraphSAINTEdgeSampler, GraphSAINTRandomWalkSampler
from torch_geometric.data import GraphSAINTNodeSampler, Data


def train_saint(model, optimizer, g, feats, labels, mask=None, epochs=1, weights=None, sampling='node', walk_length=2,
                coverage=200, batch_size=1000, l=0.1, prob_exp=1):
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')
    cpu = th.device('cpu')

    sampler_args = {'data': Data(x=feats.to(cpu), edge_index=g.to(cpu), y=labels), 'batch_size': batch_size,
                    'num_workers': 0, 'num_steps': epochs, 'sample_coverage': coverage}
    if sampling == "node":
        sampler = GraphSAINTNodeSampler
    elif sampling == "edge":
        sampler = GraphSAINTEdgeSampler
    elif sampling == "rw":
        sampler = GraphSAINTRandomWalkSampler
        sampler_args['walk_length'] = walk_length
    else:
        raise NotImplementedError(f"\"{sampling}\" is not a supported sampling method for GraphSAINT!")
    model.train()

    for i, data in enumerate(sampler(**sampler_args)):
        # mask -> saint sampled subgraph
        data = data.to(device)
        model = model.to(device)
        # Debug
        # print("X",data.x.size())
        # print("edge_index",data.edge_index.size())
        logits = model(data.x, data.edge_index, data.edge_norm)

        logits = logits.to(cpu)
        data = data.to(cpu)

        if mask is not None:
            loss = F.cross_entropy(logits[mask], labels[mask])
            loss = (loss * data.node_norm[mask])
        else:
            loss = F.cross_entropy(logits, data.y)
            loss = (loss * data.node_norm)

        if weights is not None:
            loss = loss * weights

        loss = loss.sum()



        # Step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("Step {:d} | Loss: {:.4f}".format(i, loss.detach().item()))


def evaluate_saint(model, g, feats, labels, mask=None, compute_loss=True, f1=False):
    model.eval()
    with th.no_grad():
        try:
            logits = model(feats, g)
        except:
            model = model.cpu()
            feats = feats.cpu()
            labels = labels.cpu()
            g = g.cpu()
            logits = model(feats, g)

        print("Logits size pre-mask", logits.size())
        print("Mask:", mask)
        if mask is not None:
            logits = logits[mask]
            labels = labels[mask]
        print("Logits size post-mask", logits.size())
        input()

        if compute_loss:
            loss = F.cross_entropy(logits, labels).item()
        else:
            loss = None

        if isinstance(logits, np.ndarray):
            logits = th.FloatTensor(logits)
        __max_vals, max_indices = th.max(logits.detach(), 1)
        if not f1:
            acc = (max_indices == labels).sum().float() / labels.size(0)
        else:
            acc = f1_score(labels.cpu(), max_indices.cpu(), average="macro")
    return acc.item(), loss
