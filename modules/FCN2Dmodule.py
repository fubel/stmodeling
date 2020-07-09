import torch
import torch.nn as nn
import torch.nn.functional as F


class FCN2Dmodule(torch.nn.Module):
    """
    This is the FCN2Dmodule implementation used for linking spatio-temporal
    features coming from different segments.
    """

    def __init__(self, img_feature_dim, num_frames, num_class, relation_type):
        super(FCN2Dmodule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.relation_type = relation_type
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(1, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=(1, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(p=0.0),
            nn.Conv2d(256, num_class, kernel_size=1),
            nn.ReLU(),
            nn.AvgPool2d((num_frames, 8))
        )

    def forward(self, input):
        input = input.unsqueeze(1)
        # print(sum(p.numel() for p in self.classifier.parameters() if p.requires_grad))
        output = self.classifier(input)
        output = torch.squeeze(output, 3)
        output = torch.squeeze(output, 2)
        return output


def return_FCN2D(relation_type, img_feature_dim, num_frames, num_class):
    FCN2Dmodel = FCN2Dmodule(img_feature_dim, num_frames, num_class, relation_type)

    return FCN2Dmodel
