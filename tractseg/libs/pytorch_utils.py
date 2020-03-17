
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn


def save_checkpoint(path, **kwargs):
    for key, value in list(kwargs.items()):
        if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
            kwargs[key] = value.state_dict()

    torch.save(kwargs, path)


def load_checkpoint(path, **kwargs):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    for key, value in list(kwargs.items()):
        if key in checkpoint:
            if isinstance(value, torch.nn.Module) or isinstance(value, torch.optim.Optimizer):
                value.load_state_dict(checkpoint[key])
            else:
                kwargs[key] = checkpoint[key]

    return kwargs


def load_checkpoint_selectively(path, **kwargs):
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    # kwargs e.g. {"unet": a_pytorch_network}
    for key, value in list(kwargs.items()):
        # value: the pytorch network class
        if key in checkpoint:
            remove_layers = ['output_2.weight', 'output_2.bias', 'output_3.weight', 'output_3.bias',
                             'conv_5.weight', 'conv_5.bias']

            model_dict = value.state_dict()  # the new untrained model as dict
            pretrained_dict = checkpoint[key]  # the pretrained model as dict

            for k, v in pretrained_dict.items():
                if k not in remove_layers:
                    model_dict[k] = v  # this should also work!

            value.load_state_dict(model_dict)  # dict to model

    return kwargs


def f1_score_macro(y_true, y_pred, per_class=False, threshold=0.5):
    """
    Macro f1. Same results as sklearn f1 macro.

    Args:
        y_true: [bs, classes, x, y]
        y_pred: [bs, classes, x, y]

    Returns:
        f1
    """
    y_true = y_true.byte()
    y_pred = (y_pred > threshold).byte()

    if len(y_true.size()) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)

        y_true = y_true.contiguous().view(-1, y_true.size()[3])  # [bs*x*y, classes]
        y_pred = y_pred.contiguous().view(-1, y_pred.size()[3])
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

        y_true = y_true.contiguous().view(-1, y_true.size()[4])  # [bs*x*y, classes]
        y_pred = y_pred.contiguous().view(-1, y_pred.size()[4])

    f1s = []
    for i in range(y_true.size()[1]):
        intersect = torch.sum(y_true[:, i] * y_pred[:, i])  # works because all multiplied by 0 gets 0
        denominator = torch.sum(y_true[:, i]) + torch.sum(y_pred[:, i])  # works because all multiplied by 0 gets 0
        f1 = (2 * intersect.float()) / (denominator.float() + 1e-6)
        f1s.append(f1.to('cpu'))
    if per_class:
        return np.array(f1s)
    else:
        return np.mean(np.array(f1s))


def f1_score_binary(y_true, y_pred):
    """
    Binary f1. Same results as sklearn f1 binary.

    Args:
        y_true: [bs*x*y], binary
        y_pred: [bs*x*y], binary

    Returns:
        f1
    """
    intersect = torch.sum(y_true * y_pred)  # works because all multiplied by 0 gets 0
    denominator = torch.sum(y_true) + torch.sum(y_pred)  # works because all multiplied by 0 gets 0
    f1 = (2 * intersect.float()) / (denominator.float() + 1e-6)
    return f1


def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(ax)
    return input


def soft_sample_dice(net_output, gt, eps=1e-6):
    axes = tuple(range(2, len(net_output.size())))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()


def soft_batch_dice(net_output, gt, eps=1e-6):
    axes = tuple([0] + list(range(2, len(net_output.size()))))
    intersect = sum_tensor(net_output * gt, axes, keepdim=False)
    denom = sum_tensor(net_output + gt, axes, keepdim=False)
    return 1 - (2 * intersect.float() / (denom.float() + eps)).mean()


def MSE_weighted(y_pred, y_true, weights):
    loss = weights * ((y_pred - y_true) ** 2)
    return torch.mean(loss)


def angle_last_dim(a, b):
    '''
    Calculate the angle between two nd-arrays (array of vectors) along the last dimension.
    Returns dot product without applying arccos -> higher value = lower angle

    dot product <-> degree conversion: 1->0°, 0.9->23°, 0.7->45°, 0->90°
    By using np.arccos you could return degree in pi (90°: 0.5*pi)

    return: one dimension less than input
    '''
    from tractseg.libs.pytorch_einsum import einsum

    if len(a.shape) == 4:
        return torch.abs(einsum('abcd,abcd->abc', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))
    else:
        return torch.abs(einsum('abcde,abcde->abcd', a, b) / (torch.norm(a, 2., -1) * torch.norm(b, 2, -1) + 1e-7))


def angle_loss(y_pred, y_true, weights=None):
    """
    Loss based on consine similarity.

    Does not need weighting. y_true is 0 all over background, therefore angle will also be 0 in those areas -> no
    extra masking of background needed.

    Args:
        y_pred: [bs, classes, x, y, z]
        y_true: [bs, classes, x, y, z]

    Returns:
        (loss, None)
    """
    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

    nr_of_classes = int(y_true.shape[-1] / 3.)
    scores = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[:, :, :, (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]

        angles = angle_last_dim(y_pred_bund, y_true_bund)  # range [0,1], 1 is best

        angles_weighted = angles
        scores[idx] = torch.mean(angles_weighted)

    # doing 1-angle would also work, but 1 will be removed when taking derivatives anyways -> kann simply do *-1
    return -torch.mean(scores), None  # range [0,-1], -1 is best


def angle_length_loss(y_pred, y_true, weights):
    """
    Loss based on combination of cosine similarity (angle error) and peak length (length error).
    """
    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
        weights = weights.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)
        weights = weights.permute(0, 2, 3, 4, 1)

    nr_of_classes = int(y_true.shape[-1] / 3.)
    scores = torch.zeros(nr_of_classes)
    angles_all = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[..., (idx * 3):(idx * 3) + 3].contiguous()
        y_true_bund = y_true[..., (idx * 3):(idx * 3) + 3].contiguous()  # [x,y,z,3]
        weights_bund = weights[..., (idx * 3)].contiguous()  # [x,y,z]

        angles = angle_last_dim(y_pred_bund, y_true_bund)
        angles_all[idx] = torch.mean(angles)
        angles_weighted = angles / weights_bund
        lengths = (torch.norm(y_pred_bund, 2., -1) - torch.norm(y_true_bund, 2, -1)) ** 2
        lenghts_weighted = lengths * weights_bund

        # Divide by weights.max otherwise lengths would be way bigger
        #   Would also work: just divide by inverted weights_bund
        #   -> keeps foreground the same and penalizes the background less
        #   (weights.max just simple way of getting the current weight factor
        #   (because weights_bund is tensor, but we want scalar))
        #   Flip angles to make it a minimization problem
        combined = -angles_weighted + lenghts_weighted / weights_bund.max()

        # Loss is the same as the following:
        # combined = 1/weights_bund * -angles + weights_bund/weights_factor * lengths
        # where weights_factor = weights_bund.max()
        # Note: For angles we need /weights and for length we need *weights. Because angles goes from 0 to -1 and
        # lengths goes from 100+ to 0. Division by weights factor is needed to balance angles and lengths terms relative
        # to each other.

        # The following would not work:
        # combined = weights_bund * (-angles + lengths)
        # angles and lengths are both multiplied with weights. But one needs to be multiplied and one divided.

        scores[idx] = torch.mean(combined)

    return torch.mean(scores), -torch.mean(angles_all).item()


def l2_loss(y_pred, y_true, weights=None):
    """
    Calculate the euclidian distance (=l2 norm / frobenius norm) between tensors.
    Expects a tensor image as input (6 channels per class).

    Args:
        y_pred: [bs, classes, x, y, z]
        y_true: [bs, classes, x, y, z]
        weights: None, just for keeping the interface the same for all loss functions

    Returns:
        loss
    """
    if len(y_pred.shape) == 4:  # 2D
        y_true = y_true.permute(0, 2, 3, 1)
        y_pred = y_pred.permute(0, 2, 3, 1)
    else:  # 3D
        y_true = y_true.permute(0, 2, 3, 4, 1)
        y_pred = y_pred.permute(0, 2, 3, 4, 1)

    nr_of_classes = int(y_true.shape[-1] / 6.)
    scores = torch.zeros(nr_of_classes)

    for idx in range(nr_of_classes):
        y_pred_bund = y_pred[:, :, :, (idx * 6):(idx * 6) + 6].contiguous()
        y_true_bund = y_true[:, :, :, (idx * 6):(idx * 6) + 6].contiguous()  # [x,y,z,6]

        dist = torch.dist(y_pred_bund, y_true_bund, 2)  # calc l2 norm / euclidian distance / frobenius norm
        scores[idx] = torch.mean(dist)

    return torch.mean(scores), None


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    nonlinearity = nn.LeakyReLU(inplace=True)

    if batchnorm:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    nonlinearity = nn.LeakyReLU(inplace=True)

    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer


def conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, batchnorm=False):
    nonlinearity = nn.LeakyReLU(inplace=True)

    if batchnorm:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm3d(out_channels),
            nonlinearity)
    else:
        layer = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias),
            nonlinearity)
    return layer


def deconv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=0, output_padding=0, bias=True):
    nonlinearity = nn.LeakyReLU(inplace=True)

    layer = nn.Sequential(
        nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                           padding=padding, output_padding=output_padding, bias=bias),
        nonlinearity)
    return layer
