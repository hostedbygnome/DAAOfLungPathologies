import torch as t
from torch import nn


def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = t.min(t.unsqueeze(a[:, 2], dim=1), b[:, 2]) - t.max(t.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = t.min(t.unsqueeze(a[:, 3], dim=1), b[:, 3]) - t.max(t.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = t.clamp(iw, min=0)
    ih = t.clamp(ih, min=0)

    ua = t.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = t.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU


class FocalLoss(nn.Module):
    def forward(self, classifications, regressions, anchors, annotations):
        alpha = 0.25
        gamma = 2.0
        batch_size = classifications.shape[0]
        classification_losses = []
        regression_losses = []

        anchor = anchors[0, :, :]

        anchor_w = anchor[:, 2] - anchor[:, 0]
        anchor_h = anchor[:, 3] - anchor[:, 1]
        anchor_ctr_x = anchor[:, 0] + 0.5 * anchor_w
        anchor_ctr_y = anchor[:, 1] + 0.5 * anchor_h

        for j in range(batch_size):

            classification = classifications[j, :, :]
            regression = regressions[j, :, :]

            bbox_annotation = annotations[j, :, :]
            bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1]

            classification = t.clamp(classification, 1e-4, 1.0 - 1e-4)

            if bbox_annotation.shape[0] == 0:
                if t.cuda.is_available():
                    alpha_factor = t.ones(classification.shape).cuda() * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * t.pow(focal_weight, gamma)

                    bce = -(t.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(t.tensor(0).float().cuda())

                else:
                    alpha_factor = t.ones(classification.shape) * alpha

                    alpha_factor = 1. - alpha_factor
                    focal_weight = classification
                    focal_weight = alpha_factor * t.pow(focal_weight, gamma)

                    bce = -(t.log(1.0 - classification))

                    # cls_loss = focal_weight * torch.pow(bce, gamma)
                    cls_loss = focal_weight * bce
                    classification_losses.append(cls_loss.sum())
                    regression_losses.append(t.tensor(0).float())

                continue

            IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4])  # num_anchors x num_annotations

            IoU_max, IoU_argmax = t.max(IoU, dim=1)  # num_anchors x 1

            # import pdb
            # pdb.set_trace()

            # compute the loss for classification
            targets = t.ones(classification.shape) * -1

            if t.cuda.is_available():
                targets = targets.cuda()

            targets[t.lt(IoU_max, 0.4), :] = 0

            positive_indices = t.ge(IoU_max, 0.5)

            num_positive_anchors = positive_indices.sum()

            assigned_annotations = bbox_annotation[IoU_argmax, :]

            targets[positive_indices, :] = 0
            targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

            if t.cuda.is_available():
                alpha_factor = t.ones(targets.shape).cuda() * alpha
            else:
                alpha_factor = t.ones(targets.shape) * alpha

            alpha_factor = t.where(t.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = t.where(t.eq(targets, 1.), 1. - classification, classification)
            focal_weight = alpha_factor * t.pow(focal_weight, gamma)

            bce = -(targets * t.log(classification) + (1.0 - targets) * t.log(1.0 - classification))

            # cls_loss = focal_weight * torch.pow(bce, gamma)
            cls_loss = focal_weight * bce

            if t.cuda.is_available():
                cls_loss = t.where(t.ne(targets, -1.0), cls_loss, t.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = t.where(t.ne(targets, -1.0), cls_loss, t.zeros(cls_loss.shape))

            classification_losses.append(cls_loss.sum() / t.clamp(num_positive_anchors.float(), min=1.0))

            # compute the loss for regression

            if positive_indices.sum() > 0:
                assigned_annotations = assigned_annotations[positive_indices, :]

                anchor_widths_pi = anchor_w[positive_indices]
                anchor_heights_pi = anchor_h[positive_indices]
                anchor_ctr_x_pi = anchor_ctr_x[positive_indices]
                anchor_ctr_y_pi = anchor_ctr_y[positive_indices]

                gt_widths = assigned_annotations[:, 2] - assigned_annotations[:, 0]
                gt_heights = assigned_annotations[:, 3] - assigned_annotations[:, 1]
                gt_ctr_x = assigned_annotations[:, 0] + 0.5 * gt_widths
                gt_ctr_y = assigned_annotations[:, 1] + 0.5 * gt_heights

                # clip widths to 1
                gt_widths = t.clamp(gt_widths, min=1)
                gt_heights = t.clamp(gt_heights, min=1)

                targets_dx = (gt_ctr_x - anchor_ctr_x_pi) / anchor_widths_pi
                targets_dy = (gt_ctr_y - anchor_ctr_y_pi) / anchor_heights_pi
                targets_dw = t.log(gt_widths / anchor_widths_pi)
                targets_dh = t.log(gt_heights / anchor_heights_pi)

                targets = t.stack((targets_dx, targets_dy, targets_dw, targets_dh))
                targets = targets.t()

                if t.cuda.is_available():
                    targets = targets / t.Tensor([[0.1, 0.1, 0.2, 0.2]]).cuda()
                else:
                    targets = targets / t.Tensor([[0.1, 0.1, 0.2, 0.2]])

                negative_indices = 1 + (~positive_indices)

                regression_diff = t.abs(targets - regression[positive_indices, :])

                regression_loss = t.where(
                    t.le(regression_diff, 1.0 / 9.0),
                    0.5 * 9.0 * t.pow(regression_diff, 2),
                    regression_diff - 0.5 / 9.0
                )
                regression_losses.append(regression_loss.mean())
            else:
                if t.cuda.is_available():
                    regression_losses.append(t.tensor(0).float().cuda())
                else:
                    regression_losses.append(t.tensor(0).float())

        return t.stack(classification_losses).mean(dim=0, keepdim=True), t.stack(regression_losses).mean(dim=0,
                                                                                                         keepdim=True)
