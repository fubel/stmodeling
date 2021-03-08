from torch import nn

from ops.basic_ops import ConsensusModule
from transforms import *
from torch.nn.init import normal_, constant_

import pretrainedmodels
from modules import FCN2Dmodule, FCN3Dmodule, DNDFmodule, MLPmodule, RNNmodule, TRNmodule, TSNmodule, CONVLSTMmodule, Transformermodule


class TSN(nn.Module):
    def __init__(self, num_class, args):
        super(TSN, self).__init__()
        self.modality = args.modality
        self.num_segments = args.num_segments
        self.num_motion = args.num_motion
        self.reshape = True
        self.before_softmax = True
        self.dropout = args.dropout
        self.dataset = args.dataset
        self.crop_num = 1
        self.consensus_type = args.consensus_type
        self.img_feature_dim = args.img_feature_dim  # the dimension of the CNN feature to represent each frame
        self.base_model_name = args.arch
        nhidden = 512
        print_spec = True
        new_length = None
        if not self.before_softmax and self.consensus_type != 'avg':
            raise ValueError("Only avg consensus can be used after Softmax")

        if new_length is None:
            if self.modality == "RGB":
                self.new_length = 1
            elif self.modality == "Flow":
                self.new_length = 5
            elif self.modality == "RGBFlow":
                # self.new_length = 1
                self.new_length = self.num_motion
        else:
            self.new_length = new_length
        if print_spec == True:
            print(("""
    Initializing TSN with base model: {}.
    TSN Configurations:
        input_modality:     {}
        num_segments:       {}
        new_length:         {}
        consensus_module:   {}
        dropout_ratio:      {}
        img_feature_dim:    {}
            """.format(self.base_model_name, self.modality, self.num_segments, self.new_length, self.consensus_type, self.dropout,
                       self.img_feature_dim)))

        self._prepare_base_model()

        feature_dim = self._prepare_tsn(num_class)

        if self.modality == 'Flow':
            print("Converting the ImageNet model to a flow init model")
            self.base_model = self._construct_flow_model(self.base_model)
            print("Done. Flow model ready...")
        elif self.modality == 'RGBDiff':
            print("Converting the ImageNet model to RGB+Diff init model")
            self.base_model = self._construct_diff_model(self.base_model)
            print("Done. RGBDiff model ready.")
        elif self.modality == 'RGBFlow':
            print("Converting the ImageNet model to RGB+Flow init model")
            self.base_model = self._construct_rgbflow_model(self.base_model)
            print("Done. RGBFlow model ready.")
        if self.consensus_type == 'MLP':
            self.consensus = MLPmodule.return_MLP(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                  num_class)
        elif self.consensus_type == 'TSN':
            self.consensus = TSNmodule.return_TSN(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                  num_class)
        elif self.consensus_type in ['TRNmultiscale']:
            self.consensus = TRNmodule.return_TRN(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                  num_class)
        elif self.consensus_type in ['FCN2D']:
            self.consensus = FCN2Dmodule.return_FCN2D(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                  num_class)
        elif self.consensus_type in ['FCN3D']:
            self.consensus = FCN3Dmodule.return_FCN3D(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                  num_class)
        elif self.consensus_type in ['LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU', 'GFLSTM', 'BLSTM']:
            self.consensus = RNNmodule.return_RNN(self.consensus_type, self.img_feature_dim, args.rnn_hidden_size,
                                                  self.num_segments, num_class, args.rnn_layer, args.rnn_dropout)
        elif self.consensus_type == 'DNDF':
            self.consensus = DNDFmodule.return_DNDF(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                    num_class)
        elif self.consensus_type == 'CONVLSTM':
            self.consensus = CONVLSTMmodule.return_CONVLSTM(self.consensus_type, self.img_feature_dim, args.rnn_hidden_size,
                                                            num_class, args.rnn_layer)
        elif self.consensus_type == 'Transformer':
            self.consensus = Transformermodule.return_Transformer(self.consensus_type, self.img_feature_dim, self.num_segments,
                                                     num_class, fc_dim=1024)
        else:
            self.consensus = ConsensusModule(consensus_type)

        if not self.before_softmax:
            self.softmax = nn.Softmax()

        self._enable_pbn = not args.no_partialbn
        if not args.no_partialbn:
            self.partialBN(True)

    def _prepare_tsn(self, num_class):
        if self.base_model_name == 'squeezenet1_1':
            last_Fire = getattr(self.base_model, self.base_model.last_layer_name)
            last_layer = getattr(last_Fire, 'expand3x3')
            feature_dim = last_layer.out_channels * 2  # Squeeze net concatenates two output from 3x3 and 1x1 kernel. So the output dimension should be doubled.
            if not self.consensus_type in ['CONVLSTM', 'FCN3D']:
                self.base_model.add_module('AvgPooling', nn.AvgPool2d(13, stride=1))
            self.base_model.add_module('fc', nn.Linear(feature_dim, num_class))
            self.base_model.last_layer_name = 'fc'
        elif self.base_model_name == 'BNInception':
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
            setattr(self.base_model, 'global_pool', nn.Dropout(p=0))
        else:
            feature_dim = getattr(self.base_model, self.base_model.last_layer_name).in_features
        

        if self.dropout == 0:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Linear(feature_dim, num_class))
            self.new_fc = None
        else:
            setattr(self.base_model, self.base_model.last_layer_name, nn.Dropout(p=self.dropout))
            if self.consensus_type in ['MLP', 'TRNmultiscale', 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU', 'FCN2D',
                                       'GFLSTM', 'BLSTM', 'DNDF','Transformer']:
                # set the MFFs feature dimension
                self.new_fc = nn.Linear(feature_dim, self.img_feature_dim)
            elif self.consensus_type in ['TSN']:
                # the default consensus types in TSN is avg
                self.new_fc = nn.Linear(feature_dim, num_class)
            elif self.consensus_type in ['CONVLSTM', 'FCN3D']:
                # the default consensus types in TSN is avg
                self.new_fc = nn.Conv2d(feature_dim, self.img_feature_dim, 1)


        std = 0.001
        if self.new_fc is None:
            normal_(getattr(self.base_model, self.base_model.last_layer_name).weight, 0, std)
            constant_(getattr(self.base_model, self.base_model.last_layer_name).bias, 0)
        else:
            normal_(self.new_fc.weight, 0, std)
            constant_(self.new_fc.bias, 0)

        return feature_dim

    def _prepare_base_model(self):

        if 'resnet' in self.base_model_name or 'vgg' in self.base_model_name or 'squeezenet1_1' in self.base_model_name:
            self.base_model = pretrainedmodels.__dict__[self.base_model_name](num_classes=1000, pretrained='imagenet')
            if self.base_model_name == 'squeezenet1_1':
                self.base_model = self.base_model.features
                self.base_model.last_layer_name = '12'
            else:
                self.base_model.last_layer_name = 'fc'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]

            if self.modality == 'Flow':
                self.input_mean = [0.5]
                self.input_std = [np.mean(self.input_std)]
            elif self.modality == 'RGBDiff':
                self.input_mean = [0.485, 0.456, 0.406] + [0] * 3 * self.new_length
                self.input_std = self.input_std + [np.mean(self.input_std) * 2] * 3 * self.new_length
        elif self.base_model_name == 'BNInception':
            self.base_model = pretrainedmodels.__dict__['bninception'](num_classes=1000, pretrained='imagenet')
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [104, 117, 128]
            self.input_std = [1]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        elif 'resnext101' in self.base_model_name:
            self.base_model = pretrainedmodels.__dict__[self.base_model_name](num_classes=1000, pretrained='imagenet')
            print(self.base_model)
            self.base_model.last_layer_name = 'last_linear'
            self.input_size = 224
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            if self.modality == 'Flow':
                self.input_mean = [128]
            elif self.modality == 'RGBDiff':
                self.input_mean = self.input_mean * (1 + self.new_length)
        else:
            raise ValueError('Unknown base model: {}'.format(self.base_model_name))

    def train(self, mode=True):
        """
        Override the default train() to freeze the BN parameters
        :return:
        """
        super(TSN, self).train(mode)
        count = 0
        if self._enable_pbn:
            print("Freezing BatchNorm2D except the first one.")
            for m in self.base_model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    count += 1
                    if count >= (2 if self._enable_pbn else 1):
                        m.eval()

                        # shutdown update in frozen mode
                        m.weight.requires_grad = False
                        m.bias.requires_grad = False

    def partialBN(self, enable):
        self._enable_pbn = enable

    def get_optim_policies(self):
        first_conv_weight = []
        first_conv_bias = []
        rnn_fc_weight = []
        rnn_fc_bias = []
        normal_weight = []
        normal_bias = []
        bn = []

        conv_cnt = 0
        bn_cnt = 0
        for m in self.modules():
            if isinstance(m, torch.nn.Conv3d) or isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    first_conv_weight.append(ps[0])
                    if len(ps) == 2:
                        first_conv_bias.append(ps[1])
                else:
                    normal_weight.append(ps[0])
                    if len(ps) == 2:
                        normal_bias.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                ps = list(m.parameters())
                rnn_fc_weight.append(ps[0])
                if len(ps) == 2:
                    rnn_fc_weight.append(ps[1])
            elif isinstance(m, torch.nn.LSTM) | isinstance(m, torch.nn.GRU) | isinstance(m, torch.nn.RNN):
                ps = list(m.parameters())
                rnn_fc_weight.append(ps[0])
                if len(ps) == 2:
                    rnn_fc_weight.append(ps[1])
            elif isinstance(m, torch.nn.BatchNorm3d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm1d):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.BatchNorm2d):
                bn_cnt += 1
                # later BN's are frozen
                if not self._enable_pbn or bn_cnt == 1:
                    bn.extend(list(m.parameters()))
            # There can be a problem here. 
            elif isinstance(m, torch.nn.Embedding):
                bn.extend(list(m.parameters()))
            elif isinstance(m, torch.nn.LayerNorm):
                bn.extend(list(m.parameters()))
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if self.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if self.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': rnn_fc_weight, 'lr_mult': 1, 'decay_mult': 1, 'name': "rnn_fc_weight"},
            {'params': rnn_fc_bias, 'lr_mult': 1, 'decay_mult': 0, 'name': "rnn_fc_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
        ]

    def forward(self, input):
        if self.modality == 'RGB':
            sample_len = 3 * self.new_length
        elif self.modality == 'Depth' or self.modality == 'Gray':
            sample_len = 1 * self.new_length
        else:
            sample_len = 2 * self.new_length

        if self.modality == 'RGBDiff':
            sample_len = 3 * self.new_length
            input = self._get_diff(input)

        if self.modality == 'RGBFlow':
            sample_len = 3 + 2 * self.new_length


        base_out = self.base_model(input.view((-1, sample_len) + input.size()[-2:]))
        if self.consensus_type in ['CONVLSTM', 'FCN3D'] and self.base_model_name == 'BNInception':
            base_out = base_out.view(base_out.size(0), 1024, 7, 7)
        base_out = base_out.squeeze()
        if self.dropout > 0:
            base_out = self.new_fc(base_out)

        if not self.before_softmax:
            base_out = self.softmax(base_out)
        if self.reshape:
            base_out = base_out.view((-1, self.num_segments) + base_out.size()[1:])

        output = self.consensus(base_out)
        return output.squeeze(1)

    def _get_diff(self, input, keep_rgb=False):
        input_c = 3 if self.modality in ["RGB", "RGBDiff"] else 2
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        if keep_rgb:
            new_data = input_view.clone()
        else:
            new_data = input_view[:, :, 1:, :, :, :].clone()

        for x in reversed(list(range(1, self.new_length + 1))):
            if keep_rgb:
                new_data[:, :, x, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]
            else:
                new_data[:, :, x - 1, :, :, :] = input_view[:, :, x, :, :, :] - input_view[:, :, x - 1, :, :, :]

        return new_data

    """ # There is no need now!!
    def _get_rgbflow(self, input):
        input_c = 3 + 2 * self.new_length # 3 is rgb channels, and 2 is coming for x & y channels of opt.flow
        input_view = input.view((-1, self.num_segments, self.new_length + 1, input_c,) + input.size()[2:])
        new_data = input_view.clone()
        return new_data
    """

    def _construct_rgbflow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        filter_conv2d = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))
        first_conv_idx = next(filter_conv2d)
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = torch.cat(
            (params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous(), params[0].data),
            1)  # NOTE: Concatanating might be other way around. Check it!
        new_kernel_size = kernel_size[:1] + (3 + 2 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_flow_model(self, base_model):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = list(filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules)))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (2 * self.new_length,) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()

        new_conv = nn.Conv2d(2 * self.new_length, conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convlution layer
        setattr(container, layer_name, new_conv)
        return base_model

    def _construct_diff_model(self, base_model, keep_rgb=False):
        # modify the convolution layers
        # Torch models are usually defined in a hierarchical way.
        # nn.modules.children() return all sub modules in a DFS manner
        modules = list(self.base_model.modules())
        first_conv_idx = filter(lambda x: isinstance(modules[x], nn.Conv2d), list(range(len(modules))))[0]
        conv_layer = modules[first_conv_idx]
        container = modules[first_conv_idx - 1]

        # modify parameters, assume the first blob contains the convolution kernels
        params = [x.clone() for x in conv_layer.parameters()]
        kernel_size = params[0].size()
        if not keep_rgb:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        else:
            new_kernel_size = kernel_size[:1] + (3 * self.new_length,) + kernel_size[2:]
            new_kernels = torch.cat(
                (params[0].data, params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()),
                1)
            new_kernel_size = kernel_size[:1] + (3 + 3 * self.new_length,) + kernel_size[2:]

        new_conv = nn.Conv2d(new_kernel_size[1], conv_layer.out_channels,
                             conv_layer.kernel_size, conv_layer.stride, conv_layer.padding,
                             bias=True if len(params) == 2 else False)
        new_conv.weight.data = new_kernels
        if len(params) == 2:
            new_conv.bias.data = params[1].data  # add bias if neccessary
        layer_name = list(container.state_dict().keys())[0][:-7]  # remove .weight suffix to get the layer name

        # replace the first convolution layer
        setattr(container, layer_name, new_conv)
        return base_model

    @property
    def crop_size(self):
        return self.input_size

    @property
    def scale_size(self):
        return self.input_size * 256 // 224

    def get_augmentation(self):
        if self.modality == 'RGB':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75, .66]),
                                                   GroupRandomHorizontalFlip(is_flow=False)
                                                   ])
        elif self.modality == 'Flow':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=True)])
        elif self.modality == 'RGBDiff':
            return torchvision.transforms.Compose([GroupMultiScaleCrop(self.input_size, [1, .875, .75]),
                                                   GroupRandomHorizontalFlip(is_flow=False)])
        elif self.modality == 'RGBFlow':
            return torchvision.transforms.Compose([GroupMultiScaleResize(0.2),
                                                   GroupMultiScaleRotate(20),
                                                   # GroupSpatialElasticDisplacement(),
                                                   GroupMultiScaleCrop(self.input_size,
                                                                       [1, .875,
                                                                        .75,
                                                                        .66]),
                                                   # GroupRandomHorizontalFlip(is_flow=False)
                                                   ])
