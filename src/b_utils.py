

import numpy as np
import scipy.sparse as sp
import torch
import os
import dgl
import torch
import random
import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse

from scipy.spatial import distance_matrix

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)

    return idx_map



def load_pokec_renewed(dataset, seed, local, label_number=1000):  # 1000
# /home/joyce/dataset/pokec_fairGNN path for in docker 
    if local: 
        edges = np.load('/Users/beep/Desktop/combinedPapers/dataset/pokec_BIND/region_job_' + str(dataset) + '_edges.npy')
        features = np.load('/Users/beep/Desktop/combinedPapers/dataset/pokec_BIND/region_job_' + str(dataset) + '_features.npy')
        labels = np.load('/Users/beep/Desktop/combinedPapers/dataset/pokec_BIND/region_job_' + str(dataset) + '_labels.npy')
        sens = np.load('/Users/beep/Desktop/combinedPapers/dataset/pokec_BIND/region_job_' + str(dataset) + '_sens.npy')
    else: 
        edges = np.load('/home/joyce/dataset/pokec_BIND/region_job_' + str(dataset) + '_edges.npy')
        features = np.load('/home/joyce/dataset/pokec_BIND/region_job_' + str(dataset) + '_features.npy')
        labels = np.load('/home/joyce/dataset/pokec_BIND/region_job_' + str(dataset) + '_labels.npy')
        sens = np.load('/home/joyce/dataset/pokec_BIND/region_job_' + str(dataset) + '_sens.npy')

    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    import random
    random.seed(seed)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    features = torch.FloatTensor(np.array(features))
    labels = torch.LongTensor(labels)
    sens = torch.LongTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens.to(torch.device('cuda'))

def load_income(dataset, seed, local, sens_attr="race", predict_attr="income", path="dataset/income", label_number=1000):  # 1000
    # /Users/beep/Desktop/combinedPapers/dataset/ (path when not on server?)
    # /home/joyce/dataset/pokec_fairGNN (path for on docker container)
    # ../data/income/ (path when not on server)
    # print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    # print("IDX_FEATURES_LABELS: ", idx_features_labels)
    header = list(idx_features_labels.columns)
    # print('HEADER: ', header)
    # header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
        # print('EDGES_UNORDERED: ', edges_unordered)

    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print('FEATURES: ', features)
    labels = idx_features_labels[predict_attr].values
    # print("LABELS: ", labels[0:20])
    idx = np.arange(features.shape[0])
    # print("idx: ", idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    # print('HELP: ', np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int)).reshape(edges_unordered.shape)
    # print('EDGES_unordered', edges_unordered.shape)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    print('DEBUG: ', edges)

    # print('EDGES: ', edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens.to(torch.device('cuda'))

# def load_pokec(dataset,sens_attr,predict_attr, path="...", label_number=1000,sens_number=500,seed=10,test_idx=False):
#     """Load data"""
#     # /Users/beep/Desktop/combinedPapers/dataset/pokec_fairGNN/region_job.csv
#     print('Loading {} dataset from {}'.format(dataset,path))

#     idx_features_labels = pd.read_csv(os.path.join(path,"{}.csv".format(dataset)))
#     header = list(idx_features_labels.columns)
#     header.remove("user_id")

#     header.remove(sens_attr)
#     header.remove(predict_attr)


#     features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
#     labels = idx_features_labels[predict_attr].values
    

#     # build graph
#     idx = np.array(idx_features_labels["user_id"], dtype=int)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt(os.path.join(path,"{}_relationship.txt".format(dataset)), dtype=int)

#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
#                      dtype=int).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
#                         shape=(labels.shape[0], labels.shape[0]),
#                         dtype=np.float32)
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

#     # features = normalize(features)
#     adj = adj + sp.eye(adj.shape[0])

#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(labels)
#     # adj = sparse_mx_to_torch_sparse_tensor(adj)

#     import random
#     random.seed(seed)
#     label_idx = np.where(labels>=0)[0]
#     random.shuffle(label_idx)

#     idx_train = label_idx[:min(int(0.5 * len(label_idx)),label_number)]
#     idx_val = label_idx[int(0.5 * len(label_idx)):int(0.75 * len(label_idx))]
#     if test_idx:
#         idx_test = label_idx[label_number:]
#         idx_val = idx_test
#     else:
#         idx_test = label_idx[int(0.75 * len(label_idx)):]




#     sens = idx_features_labels[sens_attr].values

#     sens_idx = set(np.where(sens >= 0)[0])
#     idx_test = np.asarray(list(sens_idx & set(idx_test)))
#     sens = torch.FloatTensor(sens)
#     idx_sens_train = list(sens_idx - set(idx_val) - set(idx_test))
#     random.seed(seed)
#     random.shuffle(idx_sens_train)
#     idx_sens_train = torch.LongTensor(idx_sens_train[:sens_number])


#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
    

#     # random.shuffle(sens_idx)

#     return adj, features, labels, idx_train, idx_val, idx_test, sens.to(torch.device('cuda'))


def load_nba_parameters_fairGNN(dataset, seed, local, sens_attr = "country", predict_attr = "SALARY", label_number=100, sens_number=50, path = "dataset/nba/", test_idx=True): 
     # /Users/beep/Desktop/combinedPapers/dataset/ (path when not on server?)
    # /home/joyce/dataset/pokec_fairGNN (path for on docker container)
    # ../data/income/ (path when not on server)
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    # print('IDX_FEATURES_LABELS: ', idx_features_labels)
    header = list(idx_features_labels.columns)
    # print('HEADER: ', header)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(os.path.join(path,"{}_edges.txt".format(dataset)), dtype=int)
        # print('EDGES_UNORDERED: ', edges_unordered)
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.7)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # print('FEATURES: ', features)
    labels = idx_features_labels[predict_attr].values
    # idx = np.arange(features.shape[0])
    idx = np.array(idx_features_labels["user_id"], dtype=int)
    # print("idx differences: ", idx)
    idx_map = {j: i for i, j in enumerate(idx)}
    # print('EDGES_unordered', edges_unordered.shape)
    # print('HELP: ', np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=int)).reshape(edges_unordered.shape)
    # print('DEBUG : ', np.array(list(map(idx_map.get, edges_unordered.flatten())),
    #                  )
    edges = edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)

    # np.array(list(map(idx_map.get, edges_unordered.flatten())))

    # print('EDGES: ', edges)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(20)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)
    print('IDX_TRAIN: ', idx_train)
    print('IDX_VAL: ', idx_val)
    print('IDX_TEST:', idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test, sens.to(torch.device('cuda'))

def load_bail(dataset, seed, sens_attr="WHITE", predict_attr="RECID", path="dataset/bail/", label_number=1000):
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)

    import random
    random.seed(seed)
    label_idx_0 = np.where(labels == 0)[0]
    label_idx_1 = np.where(labels == 1)[0]

    random.shuffle(label_idx_0)
    random.shuffle(label_idx_1)

    idx_train = np.append(label_idx_0[:min(int(0.5 * len(label_idx_0)), label_number // 2)],
                          label_idx_1[:min(int(0.5 * len(label_idx_1)), label_number // 2)])
    idx_val = np.append(label_idx_0[int(0.5 * len(label_idx_0)):int(0.75 * len(label_idx_0))],
                        label_idx_1[int(0.5 * len(label_idx_1)):int(0.75 * len(label_idx_1))])
    idx_test = np.append(label_idx_0[int(0.75 * len(label_idx_0)):], label_idx_1[int(0.75 * len(label_idx_1)):])

    sens = idx_features_labels[sens_attr].values.astype(int)
    sens = torch.FloatTensor(sens)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    # add .to(torch.device('cuda')) if on GPU
    return adj, features, labels, idx_train, idx_val, idx_test, sens.to(torch.device('cuda'))


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)