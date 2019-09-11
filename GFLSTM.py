import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


# GFLSTM module like nn.LSTM
class GFLSTM(nn.Module):
    """
        Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        num_layers: Number of recurrent layers.

        Inputs: input, (h_0, c_0)
        - **input** (seq_len, batch, input_size): tensor containing the features of the input sequence.
        - **h_0** (num_layers, batch, hidden_size): tensor containing
          the initial hidden state for each element in the batch.
        - **c_0** (num_layers, batch, hidden_size): tensor containing
          the initial cell state for each element in the batch.

        Outputs: output, (h_n, c_n)
        - **output** (seq_len, batch, hidden_size): tensor containing
        - **h_n** (num_layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
        - **c_n** (num_layers, batch, hidden_size): tensor containing the cell state for t=seq_len
    """

    def __init__(self, input_size, hidden_size, num_layers, batch_first=False, dropout=0):
        super(GFLSTM, self).__init__()
        self.batch_first = batch_first
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.l_i2h = [nn.Linear(input_size, hidden_size * 3)]
        self.l_h2h = [nn.Linear(hidden_size, hidden_size * 3)]
        self.l_wc = [nn.Linear(input_size, hidden_size)]
        for L in range(1, num_layers):
            self.l_i2h.append(nn.Linear(hidden_size, hidden_size * 3))
            self.l_h2h.append(nn.Linear(hidden_size, hidden_size * 3))
            self.l_wc.append(nn.Linear(hidden_size, hidden_size))
        self.l_wg = []
        self.l_ug = []
        self.l_uc = []
        for L in range(num_layers):
            _wg, _ug, _uc = [], [], []
            for _L in range(num_layers):
                if L == 0:
                    _wg.append(nn.Linear(input_size, hidden_size))
                else:
                    _wg.append(nn.Linear(hidden_size, hidden_size))
                _ug.append(nn.Linear(hidden_size * num_layers, hidden_size))
                _uc.append(nn.Linear(hidden_size, hidden_size))
            self.l_wg.append(_wg)
            self.l_ug.append(_ug)
            self.l_uc.append(_uc)
        # set attributes
        for L in range(num_layers):
            setattr(self, 'layer_i2h_%d' % L, self.l_i2h[L])
            setattr(self, 'layer_h2h_%d' % L, self.l_h2h[L])
            setattr(self, 'layer_wc_%d' % L, self.l_wc[L])
        for L in range(num_layers):
            for _L in range(num_layers):
                setattr(self, 'layer_wg_%d_%d' % (L, _L), self.l_wg[L][_L])
                setattr(self, 'layer_ug_%d_%d' % (L, _L), self.l_ug[L][_L])
                setattr(self, 'layer_uc_%d_%d' % (L, _L), self.l_uc[L][_L])
        self.l_drop = nn.Dropout(dropout, inplace=True)

    def forward_one_step(self, input, hidden):
        nowh, nowc = hidden
        nowH = F.torch.cat([nowh], 1)  # concate all hidden states
        nxth_list, nxtc_list = [], []
        for L in range(self.num_layers):
            if L > 0:
                input = self.l_drop(nxth_list[L - 1])  # (batch, input_size / hidden_size)
            h, c = nowh[L], nowc[L]  # (batch, hidden_size)
            i2h, h2h = self.l_i2h[L](input), self.l_h2h[L](h)  # (batch, hidden_size * 3)
            # cell gates
            i_gate, f_gate, o_gate = F.torch.split(F.sigmoid(i2h + h2h),
                                                   self.hidden_size,
                                                   dim=1)  # (batch, hidden_size)
            # global gates
            global_gates = []
            for _L in range(self.num_layers):
                global_gates.append(F.sigmoid(self.l_wg[L][_L](input) + self.l_ug[L][_L](nowH)))
            # decode in transform
            in_from_input = self.l_wc[L](input)
            for _L in range(self.num_layers):
                in_from_nowh = global_gates[_L] * self.l_uc[L][_L](nowh[_L])
                in_from_input = in_from_input + in_from_nowh
            in_from_input = F.tanh(in_from_input)
            # update cells and hidden
            _c = f_gate * c + i_gate * in_from_input
            _h = o_gate * F.tanh(_c)
            nxth_list.append(_h)
            nxtc_list.append(_c)
        # (num_layers, batch, hidden_size)
        nxth = F.torch.stack(nxth_list, dim=0)
        nxtc = F.torch.stack(nxtc_list, dim=0)
        output = nxth_list[-1]  # top hidden is output
        return output, (nxth, nxtc)

    def forward(self, input, hidden):
        if self.batch_first:  # seq_first to batch_first
            input = F.torch.stack([input], dim=1)
        output = []
        for _in in input:
            _out, hidden = self.forward_one_step(_in, hidden)
            output.append(_out)
        output = F.torch.stack(output, dim=0)
        if self.batch_first:  # seq_first to batch_first
            output = F.torch.stack(output, dim=1)
        return output, hidden

# import torch
# import torch.nn as nn
# from torch.autograd import Variable
# import torch.nn.functional as F
# # GFLSTM module like nn.LSTM
# class GFLSTM(nn.Module):
#     """
#         Args:
#         input_size: The number of expected features in the input x
#         hidden_size: The number of features in the hidden state h
#         num_layers: Number of recurrent layers.

#         Inputs: input, (h_0, c_0)
#         - **input** (seq_len, batch, input_size): tensor containing the features of the input sequence.
#         - **h_0** (num_layers, batch, hidden_size): tensor containing
#           the initial hidden state for each element in the batch.
#         - **c_0** (num_layers, batch, hidden_size): tensor containing
#           the initial cell state for each element in the batch.

#         Outputs: output, (h_n, c_n)
#         - **output** (seq_len, batch, hidden_size): tensor containing
#         - **h_n** (num_layers, batch, hidden_size): tensor containing the hidden state for t=seq_len
#         - **c_n** (num_layers, batch, hidden_size): tensor containing the cell state for t=seq_len
#     """

#     def __init__(self, input_size, hidden_size, num_layers, batch_first=True, dropout=0):
#         super(GFLSTM, self).__init__()
#         self.batch_first = batch_first
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.dropout = dropout
#         self.l_i2h = [nn.Linear(input_size, hidden_size * 3)]
#         self.l_h2h = [nn.Linear(hidden_size, hidden_size * 3)]
#         self.l_wc = [nn.Linear(input_size, hidden_size)]
#         for L in range(1, num_layers):
#             self.l_i2h.append(nn.Linear(hidden_size, hidden_size * 3))
#             self.l_h2h.append(nn.Linear(hidden_size, hidden_size * 3))
#             self.l_wc.append(nn.Linear(hidden_size, hidden_size))
#         self.l_wg = []
#         self.l_ug = []
#         self.l_uc = []
#         for L in range(num_layers):
#             _wg, _ug, _uc = [], [], []
#             for _L in range(num_layers):
#                 if L == 0:
#                     _wg.append(nn.Linear(input_size, hidden_size))
#                 else:
#                     _wg.append(nn.Linear(hidden_size, hidden_size))
#                 _ug.append(nn.Linear(hidden_size * num_layers, hidden_size))
#                 _uc.append(nn.Linear(hidden_size, hidden_size))
#             self.l_wg.append(_wg)
#             self.l_ug.append(_ug)
#             self.l_uc.append(_uc)
#         # set attributes
#         for L in range(num_layers):
#             setattr(self, 'layer_i2h_%d' % L, self.l_i2h[L])
#             setattr(self, 'layer_h2h_%d' % L, self.l_h2h[L])
#             setattr(self, 'layer_wc_%d' % L, self.l_wc[L])
#         for L in range(num_layers):
#             for _L in range(num_layers):
#                 setattr(self, 'layer_wg_%d_%d' % (L,_L), self.l_wg[L][_L])
#                 setattr(self, 'layer_ug_%d_%d' % (L,_L), self.l_ug[L][_L])
#                 setattr(self, 'layer_uc_%d_%d' % (L,_L), self.l_uc[L][_L])
#         self.l_drop = nn.Dropout(dropout, inplace=True)

#     def forward_one_step(self, input, hidden):
#         nowh, nowc = hidden
#         nowH = F.torch.cat(list(nowh), 1)  # concate all hidden states
#         nxth_list, nxtc_list = [], []
#         for L in range(self.num_layers):
#             if L > 0:
#                 input = self.l_drop(nxth_list[L - 1])  # (batch, input_size / hidden_size)
#             h, c = nowh[L], nowc[L]  # (batch, hidden_size)
#             i2h, h2h = self.l_i2h[L](input), self.l_h2h[L](h)  # (batch, hidden_size * 3)
#             # cell gates
#             i_gate, f_gate, o_gate = F.torch.split(F.sigmoid(i2h + h2h),
#                                                    self.hidden_size,
#                                                    dim=1)  # (batch, hidden_size)
#             # global gates
#             global_gates = []
#             for _L in range(self.num_layers):
#                 global_gates.append(F.sigmoid(self.l_wg[L][_L](input) + self.l_ug[L][_L](nowH)))
#             # decode in transform
#             in_from_input = self.l_wc[L](input)
#             for _L in range(self.num_layers):
#                 in_from_nowh = global_gates[_L] * self.l_uc[L][_L](nowh[_L])
#                 in_from_input = in_from_input + in_from_nowh
#             in_from_input = F.tanh(in_from_input)
#             # update cells and hidden
#             _c = f_gate * c + i_gate * in_from_input
#             _h = o_gate * F.tanh(_c)
#             nxth_list.append(_h)
#             nxtc_list.append(_c)
#         # (num_layers, batch, hidden_size)
#         nxth = F.torch.stack(nxth_list, dim=0)
#         nxtc = F.torch.stack(nxtc_list, dim=0)
#         output = nxth_list[-1]  # top hidden is output
#         return output, (nxth, nxtc)

#     def forward(self, input, hidden):
#         print("###################")
#         print(input.size())
#         if self.batch_first: # convert batch_first to seq_first
#             input = F.torch.stack(input,dim=1)
#         output = []
#         for _in in input:
#             _out, hidden = self.forward_one_step(_in, hidden)
#             output.append(_out)
#         output = F.torch.stack(output, dim=0)
#         if self.batch_first: # seq_first to batch_first
#             output = F.torch.stack(output,dim=1)
#         return output, hidden
