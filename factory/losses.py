"""
Reference:
- https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import filterfalse


########################################################################################################################
# utils
########################################################################################################################

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


class StableBCELoss(nn.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()

    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


########################################################################################################################
# binary losses (for binary or multi-label problem)

# segmentation
# input shape: (N, C, H, W)
# target shape: (N, C, H, W)

# classification
# input shape: (N, ) or (N, C)
# target shape: (N, ) or (N, C)


# shape가 같기만 하면 됩니다.
# sigmoid는 loss 안에 포함하는 걸로 통일하였습니다.
########################################################################################################################

def binary_dice_loss():
    def func(input, target):
        input = F.sigmoid(input)

        smooth = 1e-6
        intersection = (input.float() * target.float()).sum(dim=(2, 3))
        union = input.float().sum(dim=(2, 3)) + target.float().sum(dim=(2, 3))
        dice = ((2. * intersection + smooth) / (union + smooth)).mean()

        return 1 - dice

    return func


def binary_lovasz_loss():
    def func(input, target):

        loss = 0.
        for n in range(input.shape[0]):
            for c in range(input.shape[1]):
                iflat = input[n, c].view(-1)
                tflat = target[n, c].view(-1)

                loss += lovasz_hinge_flat(iflat, tflat)

        loss = loss / (input.shape[0] * input.shape[1])
        return loss

    return func


def binary_lovasz_loss_symmetric():
    def func(input, target):

        loss = 0.
        for n in range(input.shape[0]):
            for c in range(input.shape[1]):
                iflat = input[n, c].view(-1)
                tflat = target[n, c].view(-1)

                loss += (lovasz_hinge_flat(iflat, tflat) + lovasz_hinge_flat(-iflat, 1-tflat)) / 2

        loss = loss / (input.shape[0] * input.shape[1])
        return loss

    return func


class SymmetricLovaszLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SymmetricLovaszLoss, self).__init__()
    def forward(self, logits, targets, epoch):
        return ((L.lovasz_hinge(logits, targets, per_image=True))
                + (L.lovasz_hinge(-logits, 1-targets, per_image=True))) / 2


def binary_focal_loss(gamma=2):
    def func(input, target):
        assert target.size() == input.size()

        max_val = (-input).clamp(min=0)

        loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
        invprobs = F.logsigmoid(-input * (target * 2 - 1))
        loss = (invprobs * gamma).exp() * loss
        return loss.mean()

    return func


def bce():
    return nn.BCEWithLogitsLoss()


def stable_bce(ignore=None):
    def func(input, target):
        input, target = flatten_binary_scores(input, target, ignore)
        loss = StableBCELoss()(input, Variable(target.float()))
        return loss

    return func


def bce_dice(bce_weight=0.6):
    def func(input, target):
        bce_loss = bce()
        dice_loss = binary_dice_loss()

        loss = bce_weight * bce_loss(input, target) + (1-bce_weight) * dice_loss(input, target)
        return loss

    return func


########################################################################################################################
# categorical losses (for multiclass problem)

# segmentation
# input shape: (N, C)
# target shape: (N)

# classification
# input shape: (N, C)
# target shape: (N)


# nn.CrossEntropyLoss()를 기준으로 input, target shape를 맞췄습니다.
# 그래서 Softmax를 train 코드에서 직접 치셔야 하고, reshape도 해야 합니다!

# ex)
# logits = model(images)
# logits_flat = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)
# logits_flat = nn.Softmax(dim=-1)(logits_flat)
########################################################################################################################

def dice_loss():
    def func(input, target):
        input = nn.Softmax(dim=-1)(input)
        target = one_hot_embedding(target, input.shape[1])

        smooth = 1e-6
        intersection = input.float() * target.float()
        union = input.float() + target.float()
        dice = ((2. * intersection + smooth) / (union + smooth)).mean()

        return 1 - dice

    return func


def lovasz_loss():
    def func(input, target):
        input = nn.Softmax(dim=-1)(input)

        return lovasz_softmax_flat(input, target, only_present=False)

    return func


def focal_loss(gamma=2):
    def func(input, target):
        input = nn.Softmax(dim=-1)(input)
        target = one_hot_embedding(target, input.shape[1])

        L = (-target) * (1 - input) ** gamma * input.clamp(min=1e-8, max=1.0).log()
        L = L.sum(dim=1)
        L = L.mean()

        return L

    return func


def cross_entropy_loss():
    return nn.CrossEntropyLoss()


def get_loss(loss_name):
    print('loss name:', loss_name)
    f = globals().get(loss_name)
    return f()


def binary_lovasz_loss2():
    def func(input, target):

        loss = 0.
        for n in range(input.shape[0]):
            for c in range(input.shape[1]):
                iflat = input[n, c].view(-1)
                tflat = target[n, c].view(-1)

                loss += lovasz_hinge_flat(iflat, tflat)

        loss = loss / (input.shape[0] * input.shape[1])
        return loss

    return func


if __name__ == '__main__':
    import time

    input = torch.rand(8, 4, 256, 512).cuda()
    target = torch.rand(8, 4, 256, 512).cuda()

    print(binary_lovasz_loss()(input, target).item())
    print(binary_lovasz(per_image=True)(input, target).item())
    print(binary_lovasz2(per_image=False)(input, target).item())


    # start = time.time()
    # for i in range(100):
        # out = binary_lovasz_loss()(input, target)
        # out = binary_lovasz()(input, target)
        # out = binary_lovasz2()(input, target)
    # print(time.time() - start)
