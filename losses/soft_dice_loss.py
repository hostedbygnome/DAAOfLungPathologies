import torch.nn as nn


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=None):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, target):
