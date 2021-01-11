import numpy as np
import scipy.sparse as sp
import torch

def encode_onehot(labels):
    # The classes must be sorted before encoding to enable s...tatic class encoding.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path = 'data/cora/', dataset = 'cora'):
    print('Loading dataset')
    
    # get idx, features and labels
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype = np.dtype(str))
    idx = np.array(idx_features_labels[:, 0], dtype = np.int32)
    idx_map = {j: i for i, j in enumerate(idx)} # just to map 'names'
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype = np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype = np.int32)
    edges = np.array(list(idx_map.get(obj) for obj in edges_unordered.flatten()), dtype = np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
    # build symmetric adjacency matrix, multiply point-wise
    # adj.T > adj gets matrix with elements that adj.T has and adj doesn't
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj) + sp.eye(adj.shape[0])
    
    # normalize 
    features = normalize_sparse(features)
    adj = normalize_sparse(adj)
    
    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    
    idx_train = torch.LongTensor(range(140)) # 20 nodes per class, 7 classes
    idx_val = torch.LongTensor(range(200, 700)) # 500 nodes validation
    idx_test = torch.LongTensor(range(700, 1700)) # 1000 nodes test
    
    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_sparse(mx):
    """ Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    
    return correct.sum() / len(labels)





