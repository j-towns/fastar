from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import itertools

from operator import mul
import numpy.random as npr

import jax.experimental.stax as stax
import jax.lax as lax
import jax.numpy as np
from jax.scipy.special import expit as sigmoid


# Initializers
def randn(stddev=1e-2, rng=npr):
    """An initializer function for random normal coefficients."""
    def init(shape):
        return rng.normal(size=shape, scale=stddev).astype('float32')
    return init

zeros = functools.partial(np.zeros, dtype='float32')
ones = functools.partial(np.ones, dtype='float32')


def l2_normalize(x, axis=None, epsilon=1e-12):
    return x / np.sqrt(
        np.maximum(np.sum(x ** 2, axis, keepdims=True), epsilon))


def _elemwise_no_params(fun, **fun_kwargs):
  init_fun = lambda rng, input_shape: (input_shape, ())
  apply_fun = lambda params, inputs, **kwargs: fun(inputs, **fun_kwargs)
  return init_fun, apply_fun

Sigmoid = _elemwise_no_params(sigmoid)
Tanh = stax.Tanh

def GeneralConv(dimension_numbers, out_chan, filter_shape,
                strides=None, padding='VALID', W_init=None, b_init=randn(1e-6),
                rhs_dilation=None):
    """Layer construction function for a general convolution layer."""
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    if rhs_dilation is None:
        rhs_dilation = one

    strides = strides or one
    W_init = W_init or stax.glorot(rhs_spec.index('O'), rhs_spec.index('I'))
    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        bias_shape = tuple(itertools.dropwhile(lambda x: x == 1, bias_shape))
        W, b = W_init(rng, kernel_shape), b_init(bias_shape)
        return output_shape, (W, b)
    def apply_fun(params, inputs, **kwargs):
        W, b = params
        return lax.conv_general_dilated(inputs, W, strides, padding, one, rhs_dilation,
                                        dimension_numbers) + b
    return init_fun, apply_fun


Conv1D = functools.partial(GeneralConv, ('NTC', 'IOT', 'NTC'))  # TODO: are the OI the right way around?


def FanInProd():
    """Layer construction function for a fan-in product layer."""
    prod = lambda ls: functools.reduce(mul, ls)
    init_fun = lambda rng, input_shape: (input_shape[0], ())
    apply_fun = lambda params, inputs, rng=None: prod(inputs)
    return init_fun, apply_fun
FanInProd = FanInProd()


def calculate_receptive_field(filter_width, dilations, scalar_input,
                              initial_filter_width):
    receptive_field = (filter_width - 1) * sum(dilations) + 1
    if scalar_input:
        receptive_field += initial_filter_width - 1
    else:
        receptive_field += filter_width - 1
    return receptive_field


def skip_slice(output_width):
    """
    Slice in the time dimension, getting the last output_width elements
    """
    init_fun = lambda rng, input_shape: (input_shape, ())
    def apply_fun(params, inputs, **kwargs):
        skip_cut = inputs.shape[1] - output_width
        slice_sizes = [inputs.shape[0], output_width, inputs.shape[2]]
        return lax.dynamic_slice(inputs,
                                 [0, skip_cut, 0],
                                 slice_sizes)
    return init_fun, apply_fun

def combine_out_and_skip():
    """
    Add the transformed output of the resblock to the sliced input
    """
    init_fun = lambda rng, input_shape: (input_shape[0], ())  # maybe?
    def apply_fun(params, inputs, **kwargs):
        out, skip = inputs[0]
        data = inputs[1]
        out_len = out.shape[1]
        sliced_inputs = lax.dynamic_slice(data,
                                          [0, data.shape[1] - out_len, 0],
                                          [data.shape[0], out_len, data.shape[2]])
        return [sum(out, sliced_inputs), skip]
    return init_fun, apply_fun
combine_out_and_skip = combine_out_and_skip()


def dilated_causal_conv(out_channels, filter_shape, dilation):
    """
    1D conv with a dilation
    """
    # TODO:  may require a shift to make it "causal"
    return Conv1D(out_channels, filter_shape, rhs_dilation=dilation)


def resblock_layer(dilation_channels, residual_channels,
                   filter_width, dilation, output_width):
    """
    From original doc string:

    The layer contains a gated filter that connects to dense output
    and to a skip connection:

           |-> [gate]   -|        |-> 1x1 conv -> skip output
           |             |-> (*) -|
    input -|-> [filter] -|        |-> 1x1 conv -|
           |                                    |-> (+) -> dense output
           |------------------------------------|

    Where `[gate]` and `[filter]` are causal convolutions with a
    non-linear activation at the output
    """
    gate = stax.serial(dilated_causal_conv(dilation_channels,
                                           (filter_width,),
                                           (dilation,)),
                       Sigmoid)
    filter = stax.serial(dilated_causal_conv(dilation_channels,
                                             (filter_width,),
                                             (dilation,)),
                         Tanh)
    nin = Conv1D(residual_channels, (1,), padding='SAME')
    skip = stax.serial(skip_slice(output_width),
                       Conv1D(residual_channels, (1,), padding='SAME'))

    main = stax.serial(stax.FanOut(2),
                       stax.parallel(gate, filter),
                       FanInProd,
                       stax.FanOut(2),
                       stax.parallel(nin, skip))

    return stax.serial(stax.FanOut(2),
                       stax.parallel(main, stax.Identity),
                       combine_out_and_skip)


def ResblockWrapped(dilation_channels, residual_channels,
                    filter_width, dilation, output_width):
    """Wrap the resblock layer such that we add the contributions from
    the skip out and pass to the next layer along with the so called hidden state"""
    res_init, res_apply = resblock_layer(dilation_channels, residual_channels,
                                         filter_width, dilation, output_width)
    def init_fun(rng, input_shapes):
        shape1, _ = input_shapes
        _, params = res_init(rng, shape1)
        return input_shapes, params
    def apply_fun(params, inputs, **kwargs):
        hidden, out = inputs
        hidden_new, out_partial = res_apply(params, hidden)
        out_new = out + out_partial
        return hidden_new, out_new
    return init_fun, apply_fun


def keep_element(n):
    """Keep only the nth element of the input"""
    def init_fun(rng, input_shapes):
        return input_shapes[n], ()
    def apply_fun(params, inputs, **kwargs):
        return inputs[n]
    return init_fun, apply_fun


def zeros_layer(shape):
    """Add zeros in, with (batch_size, *shape)"""
    def init_fun(rng, input_shapes):
        N = input_shapes[0]
        return (N,) + shape, ()
    def apply_fun(params, inputs, **kwargs):
        N = inputs.shape[0]
        return np.zeros((N,) + shape, dtype='float32')
    return init_fun, apply_fun


def Wavenet(dilations, filter_width, initial_filter_width, out_width, residual_channels,
            dilation_channels, skip_channels, nr_mix, dropout_p=0.5):
    """
    :param dilations: dilations for each layer
    :param filter_width: for the resblock convs
    :param residual_channels: 1x1 conv output channels
    :param dilation_channels: gate and filter output channels
    :param skip_channels: channels before the final output
    :param initial_filter_width: for the pre processing conv
    :param dropout_p:
    """
    pre_pro = stax.serial(dilated_causal_conv(residual_channels, (initial_filter_width,), (1,)),
                          stax.FanOut(2),
                          stax.parallel(stax.Identity,
                                        zeros_layer((out_width, residual_channels))))

    res_blocks = []
    for d in dilations:
        res_blocks.append(ResblockWrapped(dilation_channels, residual_channels,
                                          filter_width, d, out_width))
    res_blocks = stax.serial(*res_blocks)

    post_pro = stax.serial(keep_element(1),
                           stax.Relu,
                           Conv1D(skip_channels, (1,)),
                           stax.Relu,
                           Conv1D(3 * nr_mix, (1,)))

    return stax.serial(pre_pro, res_blocks, post_pro)
