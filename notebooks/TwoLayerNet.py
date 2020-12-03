# implements a simple two layer neural network
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from math import pi, factorial

#torch.set_printoptions(profile="full")

class TwoLayerNet(torch.nn.Module):
    def __init__(self, D_in, width, num_tasks, activation):
        super(TwoLayerNet, self).__init__()
        self.width = width
        self.linear1 = torch.nn.Linear(D_in, width)
        self.linear2 = torch.nn.Linear(width, num_tasks)
        self.activation = activation

    def forward(self, x, idx):
        y = self.linear1(x)
        #if self.activation == 'relu':
        #    y = torch.relu(y)
        y = self.linear2(y)
        return y[:, idx]


def task_to_batch(batches, batch_size, X, Y, idx):
    assert(len(Y) % batch_size == 0)
    num_batch = int(len(Y) / batch_size)
    for i in range(num_batch):
        batch = {}
        batch['features'] = torch.from_numpy( X[i * batch_size: (i+1) * batch_size, :] ).float()
        batch['labels'] = torch.from_numpy( Y[i * batch_size: (i+1) * batch_size].T ).float()
        batch['task'] = idx
        batches.append(batch)


def eval_loss(data_list, model, criterion):
    valid_loss = 0
    for i in range(len(data_list)):
        batch = data_list[i]
        y_pred = model(batch['features'], batch['task'])
        valid_loss += criterion(y_pred, batch['labels'])
    return valid_loss / len(data_list)


def two_layer_square_loss(data_train, data_eval, args):
    # set up model and optimizer
    model = TwoLayerNet(args.D_in, args.width, args.num_tasks, args.activation)
    optimizer = optim.SGD(model.parameters(), args.lr, momentum=0.0)
    criterion = nn.MSELoss()

    for i in range(args.epochs):

        # mini-batch SGD
        for j in range(len(data_train)):
            batch = data_train[j]
            y_pred = model(batch['features'], batch['task'])
            train_loss = criterion(y_pred, batch['labels'])
            # if args.group_lasso:
            #    train_loss += args.lambda_group_lasso * torch.norm(model.linear1.weight, p='fro') / len(data_train)
            # elif args.weight_decay:
            #    train_loss += args.lambda_weight_decay * torch.norm(model.linear1.weight, p='fro')**2 /len(data_train)

            # take a gradient step
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

        # evaluate
        if i % args.print_every == 0:
            train_loss = eval_loss(data_train, model, criterion)
            valid_loss = eval_loss(data_eval, model, criterion)
            print(i, 'train', train_loss.item() , 'validation', valid_loss.item(), flush=True)
            
            print(model.linear1.weight.data, model.linear2.weight.data)

