import torch
import torch.nn as nn
from torch.autograd import Variable
from opts import parser
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np


class RNNmodule(torch.nn.Module):
    """
    This is the RNN implementation used for linking spatio-temporal
    features coming from different segments.
    """

    def __init__(self, rnn_type, img_feature_dim, num_frames, num_class, num_layers, hidden_size, dropout=0.5):
        super(RNNmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.rnn_type = rnn_type
        self.img_feature_dim = img_feature_dim
        self.fc = nn.Linear(hidden_size, self.num_class)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(input_size=self.img_feature_dim,
                                             hidden_size=self.hidden_size,
                                             num_layers=self.num_layers,
                                             dropout=self.dropout)
        elif (rnn_type == 'GFLSTM'):
            self.rnn = GFLSTM(input_size=self.img_feature_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              dropout=self.dropout)
        elif (rnn_type == 'BLSTM'):
            self.rnn = nn.LSTM(input_size=self.img_feature_dim,
                               hidden_size=self.hidden_size // 2,
                               num_layers=self.num_layers,
                               dropout=self.dropout,
                               bidirectional=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied""")
            self.rnn = nn.RNN(input_size=self.img_feature_dim,
                              hidden_size=self.hidden_size,
                              num_layers=self.num_layers,
                              nonlinearity=nonlinearity,
                              dropout=self.dropout)

    def forward(self, input):
        if (self.rnn_type == 'BLSTM'):
            input = input.permute(1, 0, 2)
            output_seq, _ = self.rnn(input)
            concatenated_output = torch.cat(
                (output_seq[0][:, self.hidden_size // 2:], output_seq[-1][:, :self.hidden_size // 2]), 1)
            last_output = self.fc(concatenated_output)
            return last_output.cuda()
        else:
            input = input.permute(1, 0, 2)
            output_seq, _ = self.rnn(input)
            last_output = self.fc(output_seq[-1])
            return last_output.cuda()


def return_RNN(relation_type, img_feature_dim, hidden_size, num_frames, num_class, num_layer, dropout):
    RNNmodel = RNNmodule(relation_type, img_feature_dim, num_frames, num_class, num_layer, hidden_size, dropout)

    return RNNmodel
