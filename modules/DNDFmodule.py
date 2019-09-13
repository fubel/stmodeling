import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


class Tree(nn.Module):
    def __init__(self, depth, n_in_feature, used_feature_rate, n_class, jointly_training=True):
        super(Tree, self).__init__()
        self.depth = depth
        self.n_leaf = 2 ** depth
        self.n_class = n_class
        self.jointly_training = jointly_training

        # used features in this tree
        n_used_feature = int(n_in_feature * used_feature_rate)
        onehot = np.eye(n_in_feature)
        using_idx = np.random.choice(np.arange(n_in_feature), n_used_feature, replace=False)
        self.feature_mask = onehot[using_idx].T
        self.feature_mask = Parameter(torch.from_numpy(self.feature_mask).type(torch.FloatTensor), requires_grad=False)
        # leaf label distribution
        self.pi = np.random.rand(self.n_leaf, n_class)
        self.pi = Parameter(torch.from_numpy(self.pi).type(torch.FloatTensor), requires_grad=True)

        # decision
        self.decision = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(n_used_feature, self.n_leaf)),
            # ('bn1', nn.BatchNorm1d(self.n_leaf)),
            ('sigmoid', nn.Sigmoid()),
        ]))

    def forward(self, x):
        """
        :param x(Variable): [batch_size,n_features]
        :return: route probability (Variable): [batch_size,n_leaf]
        """
        if x.is_cuda and not self.feature_mask.is_cuda:
            self.feature_mask = self.feature_mask.cuda()

        feats = torch.mm(x, self.feature_mask)  # ->[batch_size,n_used_feature]
        decision = self.decision(feats)  # ->[batch_size,n_leaf]

        decision = torch.unsqueeze(decision, dim=2)
        decision_comp = 1 - decision
        decision = torch.cat((decision, decision_comp), dim=2)  # -> [batch_size,n_leaf,2]

        # compute route probability
        # note: we do not use decision[:,0]
        batch_size = x.size()[0]
        _mu = Variable(x.data.new(batch_size, 1, 1).fill_(1.))
        begin_idx = 1
        end_idx = 2
        for n_layer in range(0, self.depth):
            _mu = _mu.view(batch_size, -1, 1).repeat(1, 1, 2)
            _decision = decision[:, begin_idx:end_idx, :]  # -> [batch_size,2**n_layer,2]
            _mu = _mu * _decision  # -> [batch_size,2**n_layer,2]
            begin_idx = end_idx
            end_idx = begin_idx + 2 ** (n_layer + 1)

        mu = _mu.view(batch_size, self.n_leaf)

        return mu

    def get_pi(self):
        f = F.softmax(self.pi, dim=-1)
        return f

    def cal_prob(self, mu, pi):
        """

        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi


class Forest(nn.Module):
    def __init__(self, n_tree, tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = Tree(tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training)
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        # for tree in self.trees:
        #    print(tree.get_pi())
        #    break
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree
        return prob


class NeuralDecisionForest(nn.Module):
    def __init__(self, forest, n_in_feature, num_class):
        super(NeuralDecisionForest, self).__init__()
        self.num_bottleneck = n_in_feature
        self.n_in_feature = n_in_feature
        self.shallow = nn.Sequential(
            nn.ReLU(),
            nn.Linear(n_in_feature,
                      n_in_feature),
            nn.Dropout(0.20),  # Add an extra DO if necess.
            nn.ReLU(),
            nn.Linear(n_in_feature, n_in_feature),
            nn.BatchNorm1d(n_in_feature)
        )
        self.forest = forest

    def forward(self, x):
        out = x.view(x.size(0), self.n_in_feature)
        out = self.shallow(out)
        out = self.forest(out)
        return out


def return_DNDF(relation_type, img_feature_dim, num_frames, num_class):
    n_in_feature = num_frames * img_feature_dim
    forest = Forest(n_tree=30, tree_depth=15, n_in_feature=n_in_feature,
                    tree_feature_rate=0.5, n_class=num_class,
                    jointly_training=True)
    model = NeuralDecisionForest(forest, n_in_feature, num_class=num_class)

    return model.cuda()
