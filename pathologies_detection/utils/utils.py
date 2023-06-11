import numpy as np
import torch as t
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.down_sample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, down_sample=None):
        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.down_sample = down_sample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class BBoxTransform(nn.Module):
    def __init__(self, mean=None, std=None):
        super(BBoxTransform, self).__init__()

        if mean is None:
            if t.cuda.is_available():
                self.mean = t.from_numpy((np.array([0, 0, 0, 0]).astype(np.float32))).cuda()
            else:
                self.mean = t.from_numpy((np.array([0, 0, 0, 0]).astype(np.float32)))
        else:
            self.mean = mean
        if std is None:
            if t.cuda.is_available():
                self.std = t.from_numpy((np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32))).cuda()
            else:
                self.std = t.from_numpy((np.array([0.1, 0.1, 0.2, 0.2]).astype(np.float32)))
        else:
            self.std = std

    def forward(self, boxes, deltas):
        widths = boxes[:, :, 2] - boxes[:, :, 0]
        heights = boxes[:, :, 3] - boxes[:, :, 1]

        ctr_x = boxes[:, :, 0] + 0.5 * widths
        ctr_y = boxes[:, :, 1] + 0.5 * heights

        dx = deltas[:, :, 0] * self.std[0] + self.mean[0]
        dy = deltas[:, :, 1] * self.std[1] + self.mean[1]
        dw = deltas[:, :, 2] * self.std[2] + self.mean[2]
        dh = deltas[:, :, 3] * self.std[3] + self.mean[3]

        pred_ctr_x = ctr_x + dx * widths
        pred_ctr_y = ctr_y + dy * heights
        pred_w = t.exp(dw) * widths
        pred_h = t.exp(dh) * heights

        pred_boxes_x_min = pred_ctr_x - 0.5 * pred_w
        pred_boxes_y_min = pred_ctr_y - 0.5 * pred_h
        pred_boxes_x_max = pred_ctr_x + 0.5 * pred_w
        pred_boxes_y_max = pred_ctr_y + 0.5 * pred_h

        pred_boxes = t.stack([pred_boxes_x_min, pred_boxes_y_min, pred_boxes_x_max, pred_boxes_y_max], dim=2)

        return pred_boxes


class ClipBoxes(nn.Module):
    def __init__(self, width=None, height=None):
        super(ClipBoxes, self).__init__()

    def forward(self, boxes, img):
        batch_size, num_channels, height, width = img.shape

        boxes[:, :, 0] = t.clamp(boxes[:, :, 0], min=0)
        boxes[:, :, 1] = t.clamp(boxes[:, :, 1], min=0)
        boxes[:, :, 2] = t.clamp(boxes[:, :, 2], max=width)
        boxes[:, :, 3] = t.clamp(boxes[:, :, 3], max=height)

        return boxes
