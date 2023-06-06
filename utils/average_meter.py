# def init_weight(*models):
#     for model in models:
#         for module in model.modules():
import numpy as np
import torch


class AverageMeter:
    def __init__(self):
        self.reset()
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def dice_coeff(pred, target):
    pred_bin = torch.where(pred > 0.5, 1., 0.)
    eps = 1e-7
    intersection = torch.sum(torch.mul(pred_bin, target))
    union = torch.sum(pred_bin) + torch.sum(target)
    dice = (2. * intersection) / (union + eps)
    return float(dice)


    # pred_bin = torch.where(pred > 0.5, 1., 0.).detach().numpy()
    # row_input = np.reshape(pred_bin, np.prod(pred_bin.shape))
    # row_target = np.reshape(target, np.prod(target.shape))
    #
    # input_square = sum(row_input)
    # target_square = sum(row_target)
    #
    # cross_square = 0
    #
    # for i in range(len(row_input)):
    #     if row_input[i] == row_target[i] == 1:
    #         cross_square += 1
    #
    # return 2 * cross_square / (input_square + target_square)



    # smooth = 1.
    # m1 = pred.view(2, 512, 512)
    # m2 = target.view(2, 512, 512)
    # intersection = (m1 * m2).sum()
    #
    # return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    # pred = pred.squeeze()
    # target = target.squeeze()
    # pred = pred[0]
    # target = target[0]
    # intersection = torch.logical_and(pred, target).sum()
    # union = torch.logical_or(target, target).sum()
    # dice_score = (2. * intersection) / (union + intersection + 1e-7)
    # return dice_score
