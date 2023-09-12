

import os
import sys
import pandas as pd
import dgl
from torch_geometric.utils import from_scipy_sparse_matrix
import functools
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder

import os.path as osp

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import train_test_split_edges
import math
import numpy as np
import scipy.sparse as sp
import scipy.io
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
# from libKMCUDA import kmeans_cuda
from tqdm import tqdm
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_geometric.datasets import KarateClub
from torch_scatter import scatter
import torch_sparse

import networkx as nx
import matplotlib.pyplot as plt
g_seed=39788
torch.set_num_threads(5)
np.random.seed(g_seed)
torch.manual_seed(g_seed)
# torch.use_deterministic_algorithms(True)
def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': F.rrelu
    }

    return activations[name]



def fair_metric(output, labels, sens):
    val_y = labels
    idx_s0 = sens.cpu().numpy()==0
    idx_s1 = sens.cpu().numpy()==1
    idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
    idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)

    pred_y = output
    parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))

    return parity,equality
def fair_metric_mc(output, labels, sens):
    val_y = labels
    idx_s0 = sens.cpu().numpy()==0
    idx_s1 = sens.cpu().numpy()==1

    pred_y = output
    
    parity =abs((len(np.where(pred_y[idx_s0]!=val_y[idx_s0])[0])/len(idx_s0))-(len(np.where(pred_y[idx_s1]!=val_y[idx_s1])[0])/len(idx_s1)))

    return parity


def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret
def feature_norm(features):
    feat_mean=torch.mean(features,0)
    feat_std=torch.std(features,0)
    return (features- feat_mean)/feat_std

def maximize_over_t(inter,intra):
    t=np.arange(0,1,0.05)
    cur_max=0
    optimized_t=0
    for i,val in enumerate(t):
        cand=np.absolute(len(np.where(inter < val)[0])/len(inter)-len(np.where(intra < val)[0])/len(intra))
        if cand>cur_max:
            cur_max=cand
            optimized_t=val
    return cur_max
def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

@repeat(3)
def label_classification(embeddings, y, sens, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')
    indices=range(np.shape(X)[0])
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X, Y,indices,
                                                                                   test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=5, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc=accuracy_score(y_test, y_pred)
    roc_auc=roc_auc_score(y_test, y_pred)

    if np.shape(y_pred)[1]>2:
        acc_parity=fair_metric_mc(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1),sens[indices_test])
        return {
            'roc_auc' : roc_auc,
            'accuracy' : acc,
            'F1Mi': micro,
            'F1Ma': macro,
            'acc_parity': acc_parity
        }
    else:
        parity,equality=fair_metric(np.argmax(y_pred,axis=1),np.argmax(y_test,axis=1),sens[indices_test])
        return {
            'roc_auc' : roc_auc,
            'accuracy' : acc,
            'F1Mi': micro,
            'F1Ma': macro,
            'parity': parity,
            'equality': equality
        }
@repeat(3)
def sens_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')
    indices=range(np.shape(X)[0])
    X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X, Y,indices,
                                                                                   test_size=1 - ratio)

    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=5, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred = prob_to_one_hot(y_pred)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    acc=accuracy_score(y_test, y_pred)
    roc_auc=roc_auc_score(y_test, y_pred)
    rb=0
    all_s=len(np.argmax(y_test,axis=1))
    for i,s in enumerate(np.unique(y.detach().cpu().numpy())):
        ind_s=np.where(np.argmax(y_pred,axis=1)==s)[0]
        acc=accuracy_score(y_test[ind_s], y_pred[ind_s])
        rb=rb+(float(len(ind_s))/np.shape(y_pred)[0])*acc
    return {'rb' : rb,
            'roc_auc' : roc_auc,
            'accuracy' : acc}
def link_prediction(embeddings, edges_tr, edges_t, neg_edges_tr, neg_edges_t, sens):
    print('entered lp')
    X = embeddings.detach().cpu().numpy()
    edges_tr = edges_tr.detach().cpu().numpy().T
    #edges_val = edges_val.detach().cpu().numpy().T
    edges_t = edges_t.detach().cpu().numpy().T
           
    
    X = normalize(X, norm='l2')
    
    X_tr=np.concatenate((X[edges_tr[:,0]],X[edges_tr[:,1]]),axis=1)
    y_tr=np.ones(np.shape(X_tr)[0])
    sens_tr=np.zeros(np.shape(X_tr)[0])
    sens_tr[np.where((sens[edges_tr[:,0]] != sens[edges_tr[:,1]]) == True)[0]]=1
    X_neg_tr=np.concatenate((X[neg_edges_tr[:,0]],X[neg_edges_tr[:,1]]),axis=1)
    y_neg_tr=np.zeros(np.shape(X_neg_tr)[0])
    sens_neg_tr=np.zeros(np.shape(X_neg_tr)[0])
    sens_neg_tr[np.where((sens[neg_edges_tr[:,0]] != sens[neg_edges_tr[:,1]]) == True)[0]]=1
    
    X_all_tr=np.concatenate((X_tr,X_neg_tr),axis=0)
    y_all_tr=np.concatenate((y_tr,y_neg_tr),axis=0)
    sens_all_tr=np.concatenate((sens_tr,sens_neg_tr),axis=0)
    
    indices_tr = np.arange(np.shape(X_all_tr)[0])
    import random
    seed=19
    random.seed(seed)
    random.shuffle(indices_tr)
    
    X_all_tr=X_all_tr[indices_tr,:]
    y_all_tr=y_all_tr[indices_tr]
    sens_all_tr=sens_all_tr[indices_tr]     
    sens_all_tr=torch.LongTensor(sens_all_tr)
    y_all_tr=y_all_tr.reshape(-1, 1) 
    print('train data generated wo onehot')
    onehot_encoder = OneHotEncoder(categories='auto').fit(y_all_tr)
    Y_all_tr = onehot_encoder.transform(y_all_tr).toarray().astype(np.bool)
    print('train data generated')


    X_t=np.concatenate((X[edges_t[:,0]],X[edges_t[:,1]]),axis=1)
    y_t=np.ones(np.shape(X_t)[0])
    sens_t=np.zeros(np.shape(X_t)[0])
    sens_t[np.where((sens[edges_t[:,0]] != sens[edges_t[:,1]]) == True)[0]]=1
    X_neg_t=np.concatenate((X[neg_edges_t[:,0]],X[neg_edges_t[:,1]]),axis=1)
    y_neg_t=np.zeros(np.shape(X_neg_t)[0])
    sens_neg_t=np.zeros(np.shape(X_neg_t)[0])
    sens_neg_t[np.where((sens[neg_edges_t[:,0]] != sens[neg_edges_t[:,1]]) == True)[0]]=1
    
    X_all_t=np.concatenate((X_t,X_neg_t),axis=0)
    y_all_t=np.concatenate((y_t,y_neg_t),axis=0)
    sens_all_t=np.concatenate((sens_t,sens_neg_t),axis=0)
    
    indices_t = np.arange(np.shape(X_all_t)[0])
    import random
    seed=19
    random.seed(seed)
    random.shuffle(indices_t)
    
    X_all_t=X_all_t[indices_t,:]
    y_all_t=y_all_t[indices_t]
    sens_all_t=sens_all_t[indices_t] 
        
    y_all_t=y_all_t.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(y_all_t)
    Y_all_t = onehot_encoder.transform(y_all_t).toarray().astype(np.bool)
    
   
    
    logreg = LogisticRegression(solver='liblinear')
    c = 2.0 ** np.arange(-10, 10)

    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__C=c), n_jobs=5, cv=5,
                       verbose=0)
    clf.fit(X_all_tr, Y_all_tr)

    y_pred = clf.predict_proba(X_all_t)
        
    y_pred = prob_to_one_hot(y_pred)
        
    micro = f1_score(Y_all_t, y_pred, average="micro")
    macro = f1_score(Y_all_t, y_pred, average="macro")
    acc=accuracy_score(Y_all_t, y_pred)
    roc_auc=roc_auc_score(Y_all_t, y_pred)

    if np.shape(y_pred)[1]>2:
        acc_parity=fair_metric_mc(np.argmax(y_pred,axis=1),np.argmax(Y_all_t,axis=1),sens_all_t)
        return {
            'roc_auc' : roc_auc,
            'accuracy' : acc,
            'F1Mi': micro,
            'F1Ma': macro,
            'acc_parity': acc_parity
        }
    else:
        parity,equality=fair_metric(np.argmax(y_pred,axis=1),np.argmax(Y_all_t,axis=1),torch.LongTensor(sens_all_t))
        return {
            'roc_auc' : roc_auc,
            'accuracy' : acc,
            'F1Mi': micro,
            'F1Ma': macro,
            'parity': parity,
            'equality': equality
        }
def load_fb(path, dataset):
    mat = scipy.io.loadmat(path+'/'+dataset)
    Adj=mat['A']
    feats=mat['local_info']
    
    idx_used=[]
    for i in range(np.shape(feats)[0]):
        if(0 not in feats[i,:]):
            idx_used.append(i)
    
    idx_nonused = np.asarray(list(set(np.arange(np.shape(feats)[0])).difference(set(idx_used))))
    #Sensitive attr is gender     
    sens=np.array(feats[idx_used,1]-1)
    
    feats=feats[idx_used,:]
    feats=feats[:,[0,2,3,4,5,6]]
    
    edges=np.concatenate((np.reshape(scipy.sparse.find(Adj)[0],(len(scipy.sparse.find(Adj)[0]),1)),np.reshape(scipy.sparse.find(Adj)[1],(len(scipy.sparse.find(Adj)[1]),1))),axis=1)

                         
    used_ind1 = [i for i, elem in enumerate(edges[:, 0]) if elem not in idx_nonused]
    used_ind2 = [i for i, elem in enumerate(edges[:, 1]) if elem not in idx_nonused]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges = edges[intersect_ind, :]

    idx_map = {j: i for i, j in enumerate(idx_used)}
    edges = np.array(list(map(idx_map.get, edges.flatten())),
                            dtype=int).reshape(edges.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(sens.shape[0], sens.shape[0]),
                            dtype=np.float32)
                         
    G = nx.from_scipy_sparse_matrix(adj)
    g_nx_ccs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    g_nx = max(g_nx_ccs, key=len)

    import random
    seed=19
    random.seed(seed)
    node_ids = list(g_nx.nodes())
    idx_s=node_ids
    random.shuffle(idx_s)
                         
    feats=feats[idx_s,:]
    feats=feats[:,np.where(np.std(np.array(feats),axis=0)!=0)[0]] 
    feats=torch.FloatTensor(np.array(feats,dtype=float))
    
    sens=torch.LongTensor(np.array(sens[idx_s],dtype=int))  
                         
    idx_map_n = {j: int(i) for i, j in enumerate(idx_s)}

    idx_nonused2 = np.asarray(list(set(np.arange(len(list(G.nodes())))).difference(set(idx_s))))
    used_ind1 = [i for i, elem in enumerate(edges[:, 0]) if elem not in idx_nonused2]
    used_ind2 = [i for i, elem in enumerate(edges[:, 1]) if elem not in idx_nonused2]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges = edges[intersect_ind, :]                     
    edges = np.array(list(map(idx_map_n.get, edges.flatten())),
                    dtype=int).reshape(edges.shape)     
    #edge_idx=np.arange(np.shape(edges)[0])
    #random.shuffle(edge_idx)
    #edges=edges[edge_idx,:]
    #num_edges=np.shape(edges)[0]
    #edges_train = edges[:int(0.9*num_edges),:]
    #edges_val = edges[int(0.8*num_edges):int(0.9*num_edges),:]
    #edges_test = edges[int(0.9*num_edges):,:]
    
    
    #adj = sp.coo_matrix((np.ones(edges_train.shape[0]), (edges_train[:, 0], edges_train[:, 1])),
    #                    shape=(sens.shape[0], sens.shape[0]),
    #                    dtype=np.float32)
    #degs=np.sum(adj.toarray(), axis=1)+np.ones(len(np.sum(adj.toarray(), axis=1)))
    #edges_train = torch.LongTensor(edges_train.T)
    
    #edges_val = torch.LongTensor(edges_val.T)
    #edges_test = torch.LongTensor(edges_test.T)
    return edges, feats, sens                  
                          
def load_pokec(dataset='region_job_2', sens_attr = "region", predict_attr= "I_am_working_in_field", path="", tris=False, degs=False):
    """Load data"""
    print('Loading {} dataset from {}'.format(dataset, path))
    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    
    header = list(idx_features_labels.columns)
    header.remove("user_id")
    # header.remove(sens_attr)
    header.remove(predict_attr)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    # features = np.array(idx_features_labels[header])
    labels = idx_features_labels[predict_attr].values
    sens = idx_features_labels[sens_attr].values
    #Only nodes for which label and sensitive attributes are available are utilized 
    sens_idx = set(np.where(sens >= 0)[0])
    label_idx = np.where(labels >= 0)[0]
    idx_used = np.asarray(list(sens_idx & set(label_idx)))
    idx_nonused = np.asarray(list(set(np.arange(len(labels))).difference(set(idx_used))))

    features = features[idx_used, :]
    print(features.shape)


    labels = labels[idx_used]
    sens = sens[idx_used]

    idx = np.array(idx_features_labels["user_id"], dtype=int)
    edges_unordered = np.genfromtxt(os.path.join(path, "{}_relationship.txt".format(dataset)), dtype=int)

    idx_n = idx[idx_nonused]
    idx = idx[idx_used]
    used_ind1 = [i for i, elem in enumerate(edges_unordered[:, 0]) if elem not in idx_n]
    used_ind2 = [i for i, elem in enumerate(edges_unordered[:, 1]) if elem not in idx_n]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_unordered = edges_unordered[intersect_ind, :]
    # build graph

    idx_map = {j: i for i, j in enumerate(idx)}
    edges_un = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                        dtype=int).reshape(edges_unordered.shape)

    
    adj = sp.coo_matrix((np.ones(edges_un.shape[0]), (edges_un[:, 0], edges_un[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    G = nx.from_scipy_sparse_matrix(adj)
    g_nx_ccs = (G.subgraph(c).copy() for c in nx.connected_components(G))
    g_nx = max(g_nx_ccs, key=len)

    import random
    seed=19
    random.seed(seed)
    node_ids = list(g_nx.nodes())
    idx_s=node_ids
    random.shuffle(idx_s)
    
    features=features[idx_s,:]
    features=features[:,np.where(np.std(np.array(features.todense()),axis=0)!=0)[0]] 
    
    features=torch.FloatTensor(np.array(features.todense()))





    labels=torch.LongTensor(labels[idx_s])
    
    sens=torch.LongTensor(sens[idx_s])

    features = feature_norm(features)
    features[:, header.index(sens_attr)] = sens


    labels[labels > 1] = 1
    sens[sens > 0] = 1
    idx_map_n = {j: int(i) for i, j in enumerate(idx_s)}

    idx_nonused2 = np.asarray(list(set(np.arange(len(list(G.nodes())))).difference(set(idx_s))))
    used_ind1 = [i for i, elem in enumerate(edges_un[:, 0]) if elem not in idx_nonused2]
    used_ind2 = [i for i, elem in enumerate(edges_un[:, 1]) if elem not in idx_nonused2]
    intersect_ind = list(set(used_ind1) & set(used_ind2))
    edges_un = edges_un[intersect_ind, :]
    edges = np.array(list(map(idx_map_n.get, edges_un.flatten())),
                     dtype=int).reshape(edges_un.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    edges= np.concatenate((np.reshape(scipy.sparse.find(adj)[0],(len(scipy.sparse.find(adj)[0]),1)),np.reshape(scipy.sparse.find(adj)[1],(len(scipy.sparse.find(adj)[1]),1))),axis=1)
    g_nx = nx.from_scipy_sparse_matrix(adj)
    edges = torch.LongTensor(edges.T)

    print(features.shape)
    print(labels.shape)
    print(sens.shape)

    # np.save(dataset + '_2_edges.npy', edges)
    # np.save(dataset + '_2_features.npy', features)
    # np.save(dataset + '_2_labels.npy', labels)
    # np.save(dataset + '_2_sens.npy', sens)

    return edges, features, labels, sens
    # if degs==True:
    #     return edges, features, labels, sens, np.sum(adj.toarray(), axis=1)
    # elif tris==True:
    #     all_cliques = nx.enumerate_all_cliques(g_nx)
    #     triad_cliques = [x for x in all_cliques if len(x) == 3]
    #     all_cliques = []
    #     return edges, features, labels, sens, np.asarray(triad_cliques)
    # else:
    #     return edges, features, labels, sens










def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask


aaa = 'user_id,public,completion_percentage,gender,region,AGE,I_am_working_in_field,spoken_languages_indicator,anglicky,nemecky,rusky,francuzsky,spanielsky,taliansky,slovensky,japonsky,hobbies_indicator,priatelia,sportovanie,pocuvanie hudby,pozeranie filmov,spanie,kupalisko,party,cestovanie,kino,diskoteky,nakupovanie,tancovanie,turistika,surfovanie po webe,praca s pc,sex,pc hry,stanovanie,varenie,jedlo,fotografovanie,citanie,malovanie,chovatelstvo,domace prace,divadlo,prace okolo domu,prace v zahrade,chodenie do muzei,zberatelstvo,hackovanie,I_most_enjoy_good_food_indicator,pri telke,v dobrej restauracii,pri svieckach s partnerom,v posteli,v prirode,z partnerovho bruska,v kuchyni pri stole,pets_indicator,pes,mam psa,nemam ziadne,macka,rybky,mam macku,mam rybky,vtacik,body_type_indicator,priemerna,vysportovana,chuda,velka a pekna,tak trosku pri sebe,eye_color_indicator,hnede,modre,zelene,hair_color_indicator,cierne,blond,plave,hair_type_indicator,kratke,dlhe,rovne,po plecia,kucerave,na jezka,completed_level_of_education_indicator,stredoskolske,zakladne,vysokoskolske,ucnovske,favourite_color_indicator,modra,cierna,cervena,biela,zelena,fialova,zlta,ruzova,oranzova,hneda,relation_to_smoking_indicator,nefajcim,fajcim pravidelne,fajcim prilezitostne,uz nefajcim,relation_to_alcohol_indicator,pijem prilezitostne,abstinent,nepijem,on_pokec_i_am_looking_for_indicator,dobreho priatela,priatelku,niekoho na chatovanie,udrzujem vztahy s priatelmi,vaznu znamost,sexualneho partnera,dlhodoby seriozny vztah,love_is_for_me_indicator,nie je nic lepsie,ako byt zamilovany(a),v laske vidim zmysel zivota,v laske som sa sklamal(a),preto som velmi opatrny(a),laska je zakladom vyrovnaneho sexualneho zivota,romanticka laska nie je pre mna,davam prednost realite,relation_to_casual_sex_indicator,nedokazem mat s niekym sex bez lasky,to skutocne zalezi len na okolnostiach,sex mozem mat iba s niekym,koho dobre poznam,dokazem mat sex s kymkolvek,kto dobre vyzera,my_partner_should_be_indicator,mojou chybajucou polovickou,laskou mojho zivota,moj najlepsi priatel,absolutne zodpovedny a spolahlivy,hlavne spolocensky typ,clovek,ktoreho uplne respektujem,hlavne dobry milenec,niekto,marital_status_indicator,slobodny(a),mam vazny vztah,zenaty (vydata),rozvedeny(a),slobodny,relation_to_children_indicator,v buducnosti chcem mat deti,I_like_movies_indicator,komedie,akcne,horory,serialy,romanticke,rodinne,sci-fi,historicke,vojnove,zahadne,mysteriozne,dokumentarne,eroticke,dramy,fantasy,muzikaly,kasove trhaky,umelecke,alternativne,I_like_watching_movie_indicator,doma z gauca,v kine,u priatela,priatelky,I_like_music_indicator,disko,pop,rock,rap,techno,house,hitparadovky,sladaky,hip-hop,metal,soundtracky,punk,oldies,folklor a ludovky,folk a country,jazz,klasicka hudba,opery,alternativa,trance,I_mostly_like_listening_to_music_indicator,kedykolvek a kdekolvek,na posteli,pri chodzi,na dobru noc,na diskoteke,s partnerom,vo vani,v aute,na koncerte,pri sexe,v praci,the_idea_of_good_evening_indicator,pozerat dobry film v tv,pocuvat dobru hudbu,s kamaratmi do baru,ist do kina alebo divadla,surfovat na sieti a chatovat,ist na koncert,citat dobru knihu,nieco dobre uvarit,zhasnut svetla a meditovat,ist do posilnovne,I_like_specialties_from_kitchen_indicator,slovenskej,talianskej,cinskej,mexickej,francuzskej,greckej,morske zivocichy,vegetarianskej,japonskej,indickej,I_am_going_to_concerts_indicator,ja na koncerty nechodim,zriedkavo,my_active_sports_indicator,plavanie,futbal,kolieskove korcule,lyzovanie,korculovanie,behanie,posilnovanie,tenis,hokej,basketbal,snowboarding,pingpong,auto-moto sporty,bedminton,volejbal,aerobik,bojove sporty,hadzana,skateboarding,my_passive_sports_indicator,baseball,golf,horolezectvo,bezkovanie,surfing,I_like_books_indicator,necitam knihy,o zabave,humor,hry,historicke romany,rozpravky,odbornu literaturu,psychologicku literaturu,literaturu pre rozvoj osobnosti,cestopisy,literaturu faktu,poeziu,zivotopisne a pamate,pocitacovu literaturu,filozoficku literaturu,literaturu o umeni a architekture'

print(len(aaa.split(',')))
load_pokec()