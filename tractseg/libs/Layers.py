# Copyright 2017 Division of Medical Image Computing, German Cancer Research Center (DKFZ)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import lasagne
import theano
import theano.tensor as T
from lasagne.layers.conv import BaseConvLayer
from theano.tensor import nnet
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
from lasagne.layers.conv import conv_input_length, conv_output_length
import theano.tensor.signal.pool
from lasagne.layers.pool import pool_output_length
from lasagne.layers import Layer
from tractseg.libs.ExpUtils import ExpUtils

class Conv3DLayer(BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1),
                 pad=0, untie_biases=False,
                 W=init.GlorotUniform(), b=init.Constant(0.),
                 nonlinearity=nonlinearities.rectify, flip_filters=True,
                 convolution=T.nnet.conv3d, **kwargs):
        BaseConvLayer.__init__(self, incoming, num_filters, filter_size,
                                          stride, pad, untie_biases, W, b,
                                          nonlinearity, flip_filters, n=3,
                                          **kwargs)
        self.convolution = convolution

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.pad == 'same' else self.pad
        conved = self.convolution(input, self.W,
                                  self.input_shape, self.get_W_shape(),
                                  subsample=self.stride,
                                  border_mode=border_mode,
                                  filter_flip=self.flip_filters)
        return conved


class Pool3DLayer(Layer):
    def __init__(self, incoming, pool_size, stride=None, pad=(0, 0, 0),
                 ignore_border=True, mode='max', **kwargs):
        super(Pool3DLayer, self).__init__(incoming, **kwargs)

        self.pool_size = as_tuple(pool_size, 3)

        if len(self.input_shape) != 5:
            raise ValueError("Tried to create a 3D pooling layer with "
                             "input shape %r. Expected 5 input dimensions "
                             "(batchsize, channels, 3 spatial dimensions)."
                             % (self.input_shape,))

        if stride is None:
            self.stride = self.pool_size
        else:
            self.stride = as_tuple(stride, 3)

        self.pad = as_tuple(pad, 3)

        self.ignore_border = ignore_border
        self.mode = mode

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)  # copy / convert to mutable list

        output_shape[2] = pool_output_length(input_shape[2],
                                             pool_size=self.pool_size[0],
                                             stride=self.stride[0],
                                             pad=self.pad[0],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[3] = pool_output_length(input_shape[3],
                                             pool_size=self.pool_size[1],
                                             stride=self.stride[1],
                                             pad=self.pad[1],
                                             ignore_border=self.ignore_border,
                                             )

        output_shape[4] = pool_output_length(input_shape[4],
                                             pool_size=self.pool_size[2],
                                             stride=self.stride[2],
                                             pad=self.pad[2],
                                             ignore_border=self.ignore_border,
                                             )

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        pooled = pool_3d(input,
                         ws=self.pool_size,
                         stride=self.stride,
                         ignore_border=self.ignore_border,
                         pad=self.pad,
                         mode=self.mode,
                         )
        return pooled

def pool_3d(input, **kwargs):
    """
    Wrapper function that calls :func:`theano.tensor.signal.pool_2d` either
    with the new or old keyword argument names expected by Theano.
    """
    try:
        return T.signal.pool.pool_3d(input, **kwargs)
    except TypeError:  # pragma: no cover
        # convert from new to old interface
        kwargs['ds'] = kwargs.pop('ws')
        kwargs['st'] = kwargs.pop('stride')
        kwargs['padding'] = kwargs.pop('pad')
        return T.signal.pool.pool_3d(input, **kwargs)

def soft_dice_fabian(y_pred, y_true):
    '''
    :param y_pred: softmax output of shape (num_samples, num_classes)
    :param y_true: one hot encoding of target (shape= (num_samples, num_classes))
    :return:
    '''
    intersect = T.sum(y_pred * y_true, 0)
    denominator = T.sum(y_pred, 0) + T.sum(y_true, 0)
    dice_scores = T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return dice_scores

def soft_dice_paul(idxs, marker, preds, ys):
    n_classes = len(ExpUtils.get_bundle_names())
    dice = T.constant(0)
    for cl in range(n_classes):
        pred = preds[marker, cl, :, :]
        y = ys[marker, cl, :, :]
        intersect = T.sum(pred * y)
        denominator = T.sum(pred) + T.sum(y)
        dice += T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return 1 - (dice / n_classes)

def theano_f1_score_OLD(idxs, marker, preds, ys):
    '''
    Von Paul
    '''
    n_classes = len(ExpUtils.get_bundle_names())
    dice = T.constant(0)
    for cl in range(n_classes):
        pred = preds[marker, cl, :, :]
        y = ys[marker, cl, :, :]
        pred = T.gt(pred, T.constant(0.5))
        intersect = T.sum(pred * y)
        denominator = T.sum(pred) + T.sum(y)
        dice += T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return dice / n_classes

def theano_f1_score_soft(idx, preds, ys):
    '''
    Expects shape of preds and ys: [bs*x*y, nr_classes]

    Does not do thresholding -> suitable to loss
    '''
    pred = preds[:, idx]
    y = ys[:, idx]
    intersect = T.sum(pred * y)
    denominator = T.sum(pred) + T.sum(y)
    dice = T.constant(2) * intersect / (denominator + T.constant(1e-6))
    return dice

def theano_f1_score(idx, preds, ys):
    '''
    Expects shape of preds and ys: [bs*x*y, nr_classes]
    '''
    pred = preds[:, idx]
    y = ys[:, idx]
    pred = T.gt(pred, T.constant(0.5))
    intersect = T.sum(pred * y)
    denominator = T.sum(pred) + T.sum(y)
    dice = (T.constant(2) * intersect) / (denominator + T.constant(1e-6))
    return dice

def theano_binary_dice_per_instance_and_class(y_pred, y_true, dim, first_spatial_axis=2):
    """
    valid for 2D and 3D, expects binary class labels in channel c
    y_pred is softmax output of shape (b,c,0,1) or (b,c,0,1,2)
    y_true's shape is equivalent

    Binarize y_pred in the beginning (-> therefore do not use this for loss)
    """

    #Needed to make it exactly the same as sklearn f1 -> we do binarization there as well
    y_pred = T.gt(y_pred, T.constant(0.5))

    # spatial_axes = tuple(range(first_spatial_axis,first_spatial_axis+dim,1))
    spatial_axes = (0,2,3)  #calc sum over all dimensions except dimension of classes
    # sum over spatial dimensions
    intersect = T.sum(y_pred * y_true, axis=spatial_axes)
    denominator = T.sum(y_pred, axis=spatial_axes) + T.sum(y_true, axis=spatial_axes)
    dice_scores = (T.constant(2) * intersect) / (denominator + T.constant(1e-6))

    # OLD: dices_scores has shape (batch_size, num_channels/num_classes)
    # dices_scores has shape (num_classes)
    return dice_scores

def theano_binary_dice_per_instance_and_class_for_loss(y_pred, y_true, dim, first_spatial_axis=2):
    """
    valid for 2D and 3D, expects binary class labels in channel c
    y_pred is softmax output of shape (b,c,0,1) or (b,c,0,1,2)
    y_true's shape is equivalent
    """
    spatial_axes = tuple(range(first_spatial_axis,first_spatial_axis+dim,1))
    # spatial_axes = (0,2,3)  #calc sum over all dimensions except dimension of classes
    # sum over spatial dimensions
    intersect = T.sum(y_pred * y_true, axis=spatial_axes)
    denominator = T.sum(y_pred, axis=spatial_axes) + T.sum(y_true, axis=spatial_axes)
    dice_scores = (T.constant(2) * intersect) / (denominator + T.constant(1e-6))

    # dices_scores has shape (batch_size, num_channels/num_classes)
    return dice_scores