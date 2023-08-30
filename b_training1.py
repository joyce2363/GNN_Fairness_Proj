from __future__ import division
from __future__ import print_function
from torch_geometric.utils import convert
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from src.b_utils import load_nba_try, load_bail, load_nba_parameters_fairGNN
from src.gcn import GCN
from scipy.stats import wasserstein_distance
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--dataset', type=str, default="nba", help='One dataset from income, bail, pokec1, and pokec2.')
parser.add_argument('--seed', type=int, default=10, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1000,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=62,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
dataset_name = args.dataset
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
def call_BIND_training1(): 
    def feature_norm(features):
        min_values = features.min(axis=0)[0]
        max_values = features.max(axis=0)[0]
        return 2*(features - min_values).div(max_values-min_values) - 1

    if dataset_name == 'nba':
        # print("loading:", dataset_name)
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_nba_parameters_fairGNN('nba')
        # print("adj_b_training:   ", adj)
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)
    elif dataset_name == 'bail':
        adj, features, labels, idx_train, idx_val, idx_test, sens = load_bail('bail')
        norm_features = feature_norm(features)
        norm_features[:, 0] = features[:, 0]
        features = feature_norm(features)

    # print("adj: ", adj)
    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
    # print("edge_index_b_training: ", edge_index)
    # nclass=labels.unique().shape[0]-1 # use this for the bail data set but nclass=1 for the nba dataset
    model = GCN(nfeat=features.shape[1], nhid=args.hidden, nclass=1, dropout=args.dropout)
    # model = FairGNN(nfeat = features.shape[1], args = args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def accuracy_new(output, labels):
        correct = output.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def fair_metric(pred, labels, sens):
        idx_s0 = sens==0
        idx_s1 = sens==1
        idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
        idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
        parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
        return parity.item(), equality.item()

    def train(epoch):
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, edge_index)
        # print("OUTPUT: ", output[:1])
        # print("idx_train: ", idx_train[:2])
        # print("output[idx_train]: ", output[idx_train][:2])
        # print("output size: ", output[idx_train].size())
        # print("what is the output: ", output)
        # print("input size: ", labels[idx_train].size())
        # print("this is what the output is: ", output)
        # # something is wrong with the input
        # print("len of input: ", labels.unsqueeze(1).float().size())
        # print("this is what the input is: ", labels)

        loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())

        preds = (output.squeeze() > 0).type_as(labels)
        acc_train = accuracy_new(preds[idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        if not args.fastmode:
            model.eval()
            output = model(features, edge_index)
        loss_val = F.binary_cross_entropy_with_logits(output[idx_val], labels[idx_val].unsqueeze(1).float())
        acc_val = accuracy_new(preds[idx_val], labels[idx_val])
        return loss_val.item()

    def tst():
        model.eval()
        output = model(features, edge_index)
        preds = (output.squeeze() > 0).type_as(labels)
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test], labels[idx_test].unsqueeze(1).float())
        acc_test = accuracy_new(preds[idx_test], labels[idx_test])

        print("*****************  Cost  ********************")
        print("SP cost:")
        idx_sens_test = sens[idx_test]
        idx_output_test = output[idx_test]
        print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))

        print("EO cost:")
        idx_sens_test = sens[idx_test][labels[idx_test]==1]
        idx_output_test = output[idx_test][labels[idx_test]==1]
        print(wasserstein_distance(idx_output_test[idx_sens_test==0].squeeze().cpu().detach().numpy(), idx_output_test[idx_sens_test==1].squeeze().cpu().detach().numpy()))
        print("**********************************************")

        parity, equality = fair_metric(preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(),
                                       sens[idx_test].numpy())

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

        print("Statistical Parity:  " + str(parity))
        print("Equality:  " + str(equality))


    t_total = time.time()
    final_epochs = 0
    loss_val_global = 1e10

    starting = time.time()
    for epoch in tqdm(range(args.epochs)):
        loss_mid = train(epoch)
        if loss_mid < loss_val_global:
            loss_val_global = loss_mid
            torch.save(model, 'gcn_' + dataset_name + '.pth')
            final_epochs = epoch

    torch.save(model, 'gcn_' + dataset_name + '.pth')

    ending = time.time()
    print("Time:", ending - starting, "s")
    model = torch.load('gcn_' + dataset_name + '.pth')
    tst()








