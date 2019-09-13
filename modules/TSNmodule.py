import torch
import torch.nn as nn


class TSNmodule(torch.nn.Module):
    """
    This is the 2-layer MLP implementation used for linking spatio-temporal
    features coming from different segments.
    """

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(TSNmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim

    def forward(self, input):
        output = input.mean(dim=1, keepdim=True)
        return output


def return_TSN(relation_type, img_feature_dim, num_frames, num_class):
    TSNmodel = TSNmodule(img_feature_dim, num_frames, num_class)

    return TSNmodel
