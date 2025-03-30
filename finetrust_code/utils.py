# import json
import torch
import numpy as np
# import pandas as pd
# import networkx as nx
from math import sqrt
# from scipy import sparse
from texttable import Texttable
# from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error,roc_auc_score

def tab_printer(args):  #打印表格
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable() 
    t.add_rows([["Parameter", "Value"]] +  [[k.replace("_"," ").capitalize(),args[k]] for k in keys])
    print(t.draw())

def read_graph(args):   #数据集处理
    """
    Method to read graph and create a target matrix with pooled adjacency matrix powers up to the order.
    :param args: Arguments object.
    :return edges: Edges dictionary.
    """
    edges = {}
    ecount = 0
    ncount = []
    negative_edges = []
    negative_labels = []
    positive_edges = []
    positive_labels = []

    with open (args.edge_path) as dataset:
        for edge in dataset:
            ecount += 1
            ncount.append(edge.split()[0])
            ncount.append(edge.split()[1])
            edge_values = edge.split()

            # all_edges.append(list(map(float, edge.split()[0:2])))
            # labels.append(list(map(float, edge.split()[2:])))
            if float(edge_values[2]) > 0:
                positive_edges.append(list(map(float, edge.split()[0:2])))
                positive_labels.append(list(map(float, edge.split()[2:])))
            else:
                negative_edges.append(list(map(float, edge.split()[0:2])))
                negative_labels.append(list(map(float, edge.split()[2:])))

              
    edges["positive_labels"] = np.array(positive_labels)
    edges["positive_edges"] = np.array(positive_edges)
    edges["negative_labels"] = np.array(negative_labels)
    edges["negative_edges"] = np.array(negative_edges)
    edges["ecount"] = ecount
    edges["ncount"] = len(set(ncount))
    return edges




def setup_features(args):  #预训练的特征
    """
    Setting up the node features as a numpy array.
    :param args: Arguments object.
    :return X: Node features.
    """
    feature = []
    with open(args.features_path) as vec:
        for node in vec:
            feature.append(node.split()[0:])  ########
    embedding = np.array(feature, np.float32)
    if args.normalize_embedding:
        return embedding / np.linalg.norm(embedding)
    else:
        return embedding



def calculate_auc(scores, prediction, label, edge):  
    label_vector = [i for line in label for i in range(len(line)) if line[i] == 1]
    """
    label_vector = []
    for line in label:
        if line[0] == 0.9:
            label_vector.append(0)
        if line[0] == 0.63:
            label_vector.append(1)
        if line[0] == 0.36:
            label_vector.append(2)
        if line[0] == 0.09:
            label_vector.append(3)
    """
    val, prediction_vector = torch.narrow(scores, 1, 0, len(label[0])).max(1)
    # auc = roc_auc_score(label_vector, prediction)
    auc = accuracy_score(label_vector, prediction_vector.cpu())  #精度
    f1_micro = f1_score(label_vector, prediction_vector.cpu(), average="micro") # average="weighted"
    f1_macro = f1_score(label_vector, prediction_vector.cpu(), average="macro")
    f1_weighted =f1_score(label_vector, prediction_vector.cpu(), average="weighted")
    
    mae_convert = {0:-1, 1:-0.9, 2:-0.8, 3:-0.7, 4:-0.6, 5:-0.5, 6:-0.4, 7:-0.3, 8:-0.2, 9:-0.1, 10:0.1, 11:0.2, 12:0.3, 13:0.4, 14:0.5, 15:0.6, 16:0.7, 17:0.8, 18:0.9, 19:1}    # #mae_convert = {0:-3, 1:-2, 2:-1, 3:1, 4:2, 5:3}
    label_mae = [mae_convert[a] for a in label_vector]
    prediction_mae = [mae_convert[a] for a in prediction_vector.cpu().numpy()]

    mae = mean_absolute_error(label_mae, prediction_mae)
    rmse = sqrt(mean_squared_error(label_mae, prediction_mae))
    return auc, f1_micro, f1_macro, f1_weighted, 0, mae, rmse

def score_printer(logs):
    """
    Print the performance for every 10th epoch on the test dataset.
    :param logs: Log dictionary.
    """
    t = Texttable() 
    t.add_rows([per for i, per in enumerate(logs["performance"]) if i % 10 == 0])
    print(t.draw())

def best_printer(log):
    t = Texttable()
    t.add_rows([per for per in log])
    print(t.draw())

"""
def save_logs(args, logs):

    #Save the logs at the path.
    #:param args: Arguments objects.
    #:param logs: Log dictionary.

    with open(args.log_path,"w") as f:
            json.dump(logs,f)
"""
# below borrowed from torch_geometric
@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


# @torch.jit._overload
# def maybe_num_nodes(edge_index, num_nodes=None):
#     # type: (SparseTensor, Optional[int]) -> int
#     pass


# def maybe_num_nodes(edge_index, num_nodes=None):
#     if num_nodes is not None:
#         return num_nodes
#     elif isinstance(edge_index, Tensor):
#         return int(edge_index.max()) + 1
#     else:
#         return max(edge_index.size(0), edge_index.size(1))

def structured_negative_sampling(edge_index, num_nodes=None):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.
    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    i, j = edge_index.to('cpu')
    idx_1 = i * num_nodes + j

    k = torch.randint(num_nodes, (i.size(0), ), dtype=torch.long)
    idx_2 = i * num_nodes + k

    mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.numel(), ), dtype=torch.long)
        idx_2 = i[rest] * num_nodes + tmp
        mask = torch.from_numpy(np.isin(idx_2, idx_1)).to(torch.bool)
        k[rest] = tmp
        rest = rest[mask.nonzero(as_tuple=False).view(-1)]

    return edge_index[0], edge_index[1], k.to(edge_index.device)