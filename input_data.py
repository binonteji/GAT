import pickle as pkl

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
import sys
import json
from torch_geometric.utils import to_undirected, to_networkx, add_self_loops, contains_isolated_nodes
from torch_geometric.data import Data 
from itertools import chain
from scipy.sparse import csr_matrix
import pandas as pd
from torch_geometric.datasets import Planetoid





def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)




def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



# def load_labels(dataset):

#     """
#     Load node-level labels
#     :param dataset: name of the input graph dataset
#     :return: n-dim array of node labels, used for community detection
#     """

#     if dataset == 'cora-large':
#         labels = np.loadtxt("../data/coralarge-cluster", delimiter = ' ', dtype = str)

#     elif dataset == 'sbm':
#         labels = np.repeat(range(100), 1000)

#     elif dataset == 'blogs':
#         labels = np.loadtxt("../data/blogs-cluster", delimiter = ' ', dtype = str)

#     elif dataset in ('cora', 'citeseer', 'pubmed'):
#         names = ['ty', 'ally']
#         objects = []
#         for i in range(len(names)):
#             with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
#                 if sys.version_info > (3, 0):
#                     objects.append(pkl.load(f, encoding = 'latin1'))
#                 else:
#                     objects.append(pkl.load(f))
#         ty, ally = tuple(objects)
#         test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
#         test_idx_range = np.sort(test_idx_reorder)
#         if dataset == 'citeseer':
#             # Fix citeseer dataset (there are some isolated nodes in the graph)
#             # Find isolated nodes, add them as zero-vecs into the right position
#             test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
#             ty_extended = np.zeros((len(test_idx_range_full), ty.shape[1]))
#             ty_extended[test_idx_range - min(test_idx_range), :] = ty
#             ty = ty_extended
#         labels = sp.vstack((ally, ty)).tolil()
#         labels[test_idx_reorder, :] = labels[test_idx_range, :]
#         # One-hot to integers
#         labels = np.argmax(labels.toarray(), axis = 1)

#     else:
#         raise ValueError('Error: undefined dataset!')

#     return labels


def load_data(dataset):

    """
    Load datasets
    :param dataset: name of the input graph dataset
    :return: n*n sparse adjacency matrix and n*f node features matrix
    """

    if dataset == 'wikics':
        with open('/home/netra-mobile/Desktop/SN Computer Science/ComDet/data/data.json', 'r') as f:
            data = json.load(f)

        x = torch.tensor(data['features'], dtype=torch.float)
        y = torch.tensor(data['labels'], dtype=torch.long)

        edges = [[(i, j) for j in js] for i, js in enumerate(data['links'])]
        edges = list(chain(*edges))
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_index = to_undirected(edge_index, num_nodes=x.size(0))
    
        data = Data(edge_index=edge_index, num_nodes=x.size(0), y=y, x=x)

        print(edge_index)
        print(x)
        print(y)

        # G = to_networkx(data)


        # adj = nx.to_scipy_sparse_matrix(G, format='csr')
        # features = x
        # labels = y


    elif dataset == 'email':
        edge_index = pd.read_csv('/home/netra-mobile/Desktop/SN Computer Science/ComDet/data/email-Eu-core.txt', sep=' ', header=None)
        edge_index = torch.from_numpy(edge_index.values).t().contiguous()



 


        y = pd.read_csv('/home/netra-mobile/Desktop/SN Computer Science/ComDet/data/email-Eu-core-department-labels.txt', sep=' ', header=None, usecols=[1])
        y = torch.from_numpy(y.values).view(-1)

        
        x = torch.eye(y.size(0), y.size(0))
        data = Data(edge_index=edge_index, num_nodes=y.size(0), y=y, x=x)

        
    elif dataset == 'cora':
        dataset = Planetoid(root='./data', name='Cora')
        data = dataset[0]


    elif dataset == 'citeseer':
        dataset = Planetoid(root='./data', name='Citeseer')
        data = dataset[0]

    
    elif dataset == 'pubmed':
        dataset = Planetoid(root='./data', name='Pubmed')
        data = dataset[0]



        #data.x = torch.eye(data.y.size(0), data.y.size(0))

        # G = to_networkx(data)


        # adj = nx.to_scipy_sparse_matrix(G, format='csr')
        # features = sp.identity(adj.shape[0])
        # features = torch.FloatTensor(np.array(features.todense()))
        # labels = y
        
    # elif dataset in ('cora', 'citeseer', 'pubmed'):
    #     # Load the data: x, tx, allx, graph
    #     names = ['x', 'tx', 'allx', 'graph']
    #     objects = []
    #     for i in range(len(names)):
    #         with open("data/ind.{}.{}".format(dataset, names[i]), 'rb') as f:
    #             if sys.version_info > (3, 0):
    #                 objects.append(pkl.load(f, encoding = 'latin1'))
    #             else:
    #                 objects.append(pkl.load(f))
    #     x, tx, allx, graph = tuple(objects)
    #     test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset))
    #     test_idx_range = np.sort(test_idx_reorder)
    #     if dataset == 'citeseer':
    #         # Fix citeseer dataset (there are some isolated nodes in the graph)
    #         # Find isolated nodes, add them as zero-vecs into the right position
    #         test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
    #         tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
    #         tx_extended[test_idx_range - min(test_idx_range), :] = tx
    #         tx = tx_extended
    #     features = sp.vstack((allx, tx)).tolil()
    #     features[test_idx_reorder, :] = features[test_idx_range, :]
    #     graph = nx.from_dict_of_lists(graph)
    #     adj = nx.adjacency_matrix(graph)



    # # else:
    # #     raise ValueError('Error: undefined dataset!')

    #     lbl =  load_labels(dataset)

    #     features = torch.FloatTensor(np.array(features.todense()))
    #     #labels = torch.LongTensor(np.where(lbl)[1])#[1]
    #     labels = lbl

    #     G = graph

        
    return data
    # adj, features, labels