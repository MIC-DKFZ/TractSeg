import os, sys, inspect
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
if not parent_dir in sys.path: sys.path.insert(0, parent_dir)

import lasagne
import theano
import theano.tensor as T
from lasagne.layers.conv import BaseConvLayer
from theano.tensor import nnet
# from theano.tensor.nnet.abstract_conv import AbstractConv3d_gradInputs
from lasagne import init
from lasagne import nonlinearities
from lasagne.utils import as_tuple
from lasagne.layers.conv import conv_input_length, conv_output_length
import theano.tensor.signal.pool
from lasagne.layers.pool import pool_output_length
from lasagne.layers import Layer
from libs.ExpUtils import ExpUtils

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

class InstanceNormLayer(Layer):
    """
    Instance Normalization
    This layer implements instance normalization of its inputs:
    .. math::
        y_i = \\frac{x_i - \\mu_i}{\\sqrt{\\sigma_i^2 + \\epsilon}}
    That is, each input example (instance) is normalized to zero mean
    and unit variance. In contrast to batch normalization, the mean and
    variance is usually taken per example, and not across examples,
    so the same operation can be applied during training and testing.
    During both training and testing, :math:`\\mu_i` and
    :math:`\\sigma_i^2` are defined to be the mean and variance
    of the instances :math:`i` in the current input mini-batch :math:`x`.
    The advantages of using this implementation over e.g.
    :class:`BatchNormLayer` with adapted axes arguments, are its
    independence of the input size, as no parameters are learned and stored.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape
    axes : 'auto', int or tuple of int
        The axis or axes to normalize over. If ``'auto'`` (the default),
        normalize over all axes except for the first two:
        this will normalize over all spatial dimensions for
        convolutional layers.
    epsilon : scalar
        Small constant :math:`\\epsilon` added to the variance before taking
        the square root and dividing by it, to avoid numerical problems
    **kwargs
        Any additional keyword arguments are passed to the :class:`Layer`
        superclass.
    Notes
    -----
    This layer should be inserted between a linear transformation (such as a
    :class:`DenseLayer`, or :class:`Conv2DLayer`) and its nonlinearity. The
    convenience function :func:`instance_norm` modifies an existing layer to
    insert instance normalization in front of its nonlinearity.
    See also
    --------
    instance_norm : Convenience function to apply instance normalization to a layer
    References
    ----------
    .. [1] Ulyanov, D., Vedaldi, A., & Lempitsky, V. (2016):
           Instance Normalization: The Missing Ingredient for Fast Stylization.
           https://arxiv.org/pdf/1607.08022.pdf.
    """
    def __init__(self, incoming, axes='auto', epsilon=1e-4,
                 beta=init.Constant(0), gamma=init.Constant(1), **kwargs):
        super(InstanceNormLayer, self).__init__(incoming, **kwargs)

        if axes == 'auto':
            # default: normalize over spatial dimensions only,
            # i.e. separate for each instance in the batch
            axes = tuple(range(2, len(self.input_shape)))
        elif isinstance(axes, int):
            axes = (axes,)
        self.axes = axes

        self.epsilon = epsilon

        if beta is None:
            self.beta = None
        else:
            self.beta = self.add_param(beta, (self.input_shape[1],), 'beta',
                                       trainable=True, regularizable=False)
        if gamma is None:
            self.gamma = None
        else:
            self.gamma = self.add_param(gamma, (self.input_shape[1],), 'gamma',
                                        trainable=True, regularizable=True)

    def get_output_for(self, input, **kwargs):

        mean = input.mean(self.axes, keepdims=True)
        std = T.sqrt(input.var(self.axes, keepdims=True) + self.epsilon)

        # prepare dimshuffle pattern inserting broadcastable axes as needed
        pattern = ['x' if input_axis != 1 else 0 for input_axis in range(input.ndim)]

        # apply dimshuffle pattern to all parameters
        beta = 0 if self.beta is None else self.beta.dimshuffle(pattern)
        gamma = 1 if self.gamma is None else self.gamma.dimshuffle(pattern)

        return gamma * (input - mean) / std + beta


def instance_norm(layer, **kwargs):
    """
    Apply instance normalization to an existing layer. This is a convenience
    function modifying an existing layer to include instance normalization: It
    will steal the layer's nonlinearity if there is one (effectively
    introducing the normalization right before the nonlinearity), remove
    the layer's bias if there is one (because it would be redundant), and add
    a :class:`InstanceNormLayer` and :class:`NonlinearityLayer` on top.
    Parameters
    ----------
    layer : A :class:`Layer` instance
        The layer to apply the normalization to; note that it will be
        irreversibly modified as specified above
    **kwargs
        Any additional keyword arguments are passed on to the
        :class:`InstanceNormLayer` constructor.
    Returns
    -------
    InstanceNormLayer or NonlinearityLayer instance
        A instance normalization layer stacked on the given modified `layer`,
        or a nonlinearity layer stacked on top of both
        if `layer` was nonlinear.
    Examples
    --------
    Just wrap any layer into a :func:`instance_norm` call on creating it:
    >>> from lasagne.layers import InputLayer, Conv2DLayer, instance_norm
    >>> from lasagne.nonlinearities import rectify
    >>> l1 = InputLayer((10, 3, 28, 28))
    >>> l2 = instance_norm(Conv2DLayer(l1, num_filters=64, filter_size=3,
    nonlinearity=rectify))
    This introduces instance normalization right before its nonlinearity:
    >>> from lasagne.layers import get_all_layers
    >>> [l.__class__.__name__ for l in get_all_layers(l2)]
    ['InputLayer', 'Conv2DLayer', 'InstanceNormLayer', 'NonlinearityLayer']
    """
    nonlinearity = getattr(layer, 'nonlinearity', None)
    if nonlinearity is not None:
        layer.nonlinearity = nonlinearities.identity

    learn_bias = True

    if 'beta' in kwargs:
        if kwargs['beta'] is None:
            learn_bias = False

    if hasattr(layer, 'b') and layer.b is not None and learn_bias:
        del layer.params[layer.b]
        layer.b = None
    in_name = (kwargs.pop('name', None) or
               (getattr(layer, 'name', None) and layer.name + '_in'))
    layer = InstanceNormLayer(layer, name=in_name, **kwargs)
    if nonlinearity is not None:
        from lasagne.layers.special import NonlinearityLayer #Manually adapted
        nonlin_name = in_name and in_name + '_nonlin'
        layer = NonlinearityLayer(layer, nonlinearity, name=nonlin_name)
    return layer