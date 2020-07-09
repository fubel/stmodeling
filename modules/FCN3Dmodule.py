import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN3Dmodule(torch.nn.Module):
    """
    This is the FCN3Dmodule implementation used for linking spatio-temporal
    features coming from different segments.
    """

    def __init__(self, img_feature_dim, num_frames, num_class, relation_type):
        super(FCN3Dmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.relation_type = relation_type
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(img_feature_dim, 64, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 1, 1), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv3d(256, num_class, kernel_size=1),
            nn.ReLU(),
        )

    def forward(self, input):
        input = input.permute(0, 2, 1, 3, 4)
        # print(sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
        output = self.classifier(input)
        output = F.avg_pool3d(output, output.data.size()[-3:]).squeeze()
        return output


def return_FCN3D(relation_type, img_feature_dim, num_frames, num_class):
    FCN3Dmodel = FCN3Dmodule(img_feature_dim, num_frames, num_class, relation_type)

    return FCN3Dmodel
