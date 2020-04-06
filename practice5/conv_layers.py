from builtins import range
import numpy as np


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, _, input_h, input_w = x.shape
    filter_amount, _, filter_h, filter_w = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    # Obtain activation map spatial size
    out_h = 1 + (input_h - filter_h + 2 * pad) // stride
    out_w = 1 + (input_w - filter_w + 2 * pad) // stride

    out = np.zeros((N, filter_amount, out_h, out_w))

    # Add padding
    padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(x, padding, mode='constant')

    for i in range(N):
      X = x_pad[i, :, :, :]

      for f in range(filter_amount):
        kernel = w[f, :, :, :]

        # Perform convolution
        for oh in range(out_h):
          h_start = oh * stride
          h_end = h_start + filter_h

          for ow in range(out_w):
            w_start = ow * stride
            w_end = w_start + filter_w

            # Convolve slice of image with kernel
            im = X[:, h_start:h_end, w_start:w_end]
            out[i, f, oh, ow] = np.sum(kernel * im) + b[f]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache

    N, _, input_h, input_w = x.shape
    filter_amount, _, filter_h, filter_w = w.shape
    stride, pad = conv_param['stride'], conv_param['pad']

    out_h = 1 + (input_h - filter_h + 2 * pad) // stride
    out_w = 1 + (input_w - filter_w + 2 * pad) // stride

    # Compute bias for each filter
    db = np.sum(dout, axis=(0, 2, 3))
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)

    # Add padding to input
    padding = ((0, 0), (0, 0), (pad, pad), (pad, pad))
    x_pad = np.pad(x, padding, mode='constant')

    # Input gradients with padding
    dx_pad = np.zeros_like(x_pad)

    for oh in range(out_h):
      h_start = oh * stride
      h_end = h_start + filter_h

      for ow in range(out_w):
        w_start = ow * stride
        w_end = w_start + filter_w

        x_masked = x_pad[:, :, h_start:h_end, w_start:w_end]

        # Convolve input using activation map to produce kernel gradients
        for k in range(filter_amount):
          gradients_filter = dout[:, k, oh, ow][:, None, None, None]
          dw[k, :, :, :] += \
            np.sum(x_masked * gradients_filter, axis=0)

        # Convolve activation map using kernel to produce input gradients
        for i in range(N):
          prod = w[:, :, :, :] * dout[i, :, oh, ow][:, None, None, None]
          dx_pad[i, :, h_start:h_end, w_start:w_end] += np.sum(prod, axis=0)

    # Remove paddings from gradients with respect to input
    dx = dx_pad[:, :, pad:-pad, pad:-pad]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    N, depth, input_h, input_w = x.shape
    pool_h, pool_w, stride = \
      pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    out_h = (input_h - pool_h) // stride + 1
    out_w = (input_w - pool_w) // stride + 1

    out = np.zeros((N, depth, out_h, out_w))

    for i in range(N):
      for oh in range(out_h):
        h_start = oh * stride
        h_end = oh * stride + pool_h

        for ow in range(out_w):
          w_start = ow * stride
          w_end = ow * stride + pool_w

          x_pool = x[i, :, h_start:h_end, w_start:w_end]
          out[i, :, oh, ow] = np.max(x_pool, axis=(1, 2))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    x, pool_param = cache

    N, depth, input_h, input_w = x.shape
    pool_h, pool_w, stride = \
      pool_param['pool_height'], pool_param['pool_width'], pool_param['stride']

    out_h = (input_h - pool_h) // stride + 1
    out_w = (input_w - pool_w) // stride + 1

    dx = np.zeros_like(x)

    for i in range(N):
      for oh in range(out_h):
        h_start = oh * stride
        h_end = oh * stride + pool_h

        for ow in range(out_w):
          w_start = ow * stride
          w_end = ow * stride + pool_w

          x_pool = x[i, :, h_start:h_end, w_start:w_end]
          max_x = np.max(x_pool, axis=(1, 2), keepdims=True)

          # Gradient with respect to input will only include the value of neuron,
          # which was activated during forward pass, else is zero.
          # No need to propagate on the non-activated values and they bring
          # no influence on the cost function
          neuron_activation_mask = x_pool == max_x
          dx[i, :, h_start:h_end, w_start:w_end] += \
            neuron_activation_mask * dout[i, :, oh, ow][:, None, None]
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx
