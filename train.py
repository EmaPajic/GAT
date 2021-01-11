import os
import glob
import numpy as np
import argparse
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from utils import load_data, accuracy
from model import GAT

def run_test(model, features, adj, idx_test, labels):
    model.eval()
    output = model(features, adj)
    loss_test = F.cross_entropy(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

def train(epoch, model, optimizer, features, adj, idx_train, idx_val):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    
    loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    
    loss_train.backward()
    optimizer.step()
    
    # validation
    model.eval()
    output = model(features, adj)
    
    loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.data.item()),
          'acc_train: {:.4f}'.format(acc_train.data.item()),
          'loss_val: {:.4f}'.format(loss_val.data.item()),
          'acc_val: {:.4f}'.format(acc_val.data.item()),
          'time: {:.4f}s'.format(time.time() - t))
    
    return loss_val.data.item()
    

if __name__ == '__main__':
    # parse training arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type = int, default = 10000, help = 'Number of epochs to train.')
    parser.add_argument('--lr', type = float, default = 0.005, help = 'Initial learning rate.')
    parser.add_argument('--weight_decay', type = float, default = 5e-4, help = 'Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type = int, default = 8, help = 'Number of hidden units.')
    parser.add_argument('--n_heads', type = int, default = 8, help = 'Number of head attentions.')
    parser.add_argument('--dropout', type = float, default = 0.6, help = 'Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type = float, default = 0.2, help = 'Alpha for the leaky_relu.')
    parser.add_argument('--patience', type = int, default = 100, help = 'Patience')

    args = parser.parse_args()
    args.use_cuda = torch.cuda.is_available()
    
    # load data
    adj, features, labels, idx_train, idx_val, idx_test = load_data()
    
    model = GAT(n_input = features.shape[1], n_hidden = args.hidden,
                n_classes = int(labels.max()) + 1, dropout = args.dropout,
                alpha = args.alpha, n_heads = args.n_heads)
    
    
    if args.use_cuda:
        model.cuda()
        features = features.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        
    optimizer = optim.Adam(model.parameters(), lr = args.lr,
                           weight_decay = args.weight_decay)
        
    features, adj, labels = Variable(features), Variable(adj), Variable(labels)
    
    # train
    
    start_time = time.time()
    loss_values = []
    patience_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    
    for epoch in range(args.epochs):
        loss_values.append(train(epoch, model, optimizer, features, adj, idx_train, idx_val))
        
        torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter == args.patience:
            break
        
        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
    
    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)
    
    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
    
    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))
      
    # Testing
    run_test(model, features, adj, idx_test, labels)