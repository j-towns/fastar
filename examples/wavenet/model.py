from jax import lax, numpy as np
from jaxnet import Sequential, parametrized, relu, sigmoid, Conv1D

def calculate_receptive_field(filter_width, dilations, scalar_input,
                              initial_filter_width):
    return ((filter_width - 1) * sum(dilations) + 1 +
            (initial_filter_width if scalar_input else filter_width) - 1)


def skip_slice(inputs, output_width):
    """Slice in the time dimension, getting the last output_width elements"""
    skip_cut = inputs.shape[1] - output_width
    slice_sizes = [inputs.shape[0], output_width, inputs.shape[2]]
    return lax.dynamic_slice(inputs, (0, skip_cut, 0), slice_sizes)


def ResBlock(dilation_channels, residual_channels,
             filter_width, dilation, output_width):
    @parametrized
    def res_layer(
            inputs,
            gate=Sequential(Conv1D(dilation_channels, (filter_width,),
                                         dilation=(dilation,)), sigmoid),
            filter=Sequential(Conv1D(dilation_channels, (filter_width,),
                                           dilation=(dilation,)), np.tanh),
            nin=Conv1D(residual_channels, (1,), padding='SAME'),
            skip_conv=Conv1D(residual_channels, (1,), padding='SAME')):
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
        p = gate(inputs) * filter(inputs)
        out = nin(p)
        # Add the transformed output of the resblock to the sliced input:
        sliced_inputs = lax.dynamic_slice(
            inputs, [0, inputs.shape[1] - out.shape[1], 0],
            [inputs.shape[0], out.shape[1], inputs.shape[2]])
        return (sum(out, sliced_inputs),
                skip_conv(skip_slice(inputs, output_width)))

    @parametrized
    def res_block(input, res_layer=res_layer):
        """Wrap the layer such that we add the contributions from the skip out
           and pass to the next layer along with the so called hidden state"""
        hidden, out = input
        hidden, out_partial = res_layer(hidden)
        return hidden, out + out_partial

    return res_block


def Wavenet(dilations, filter_width, initial_filter_width, out_width,
            residual_channels, dilation_channels, skip_channels, nr_mix):
    """
    :param dilations: dilations for each layer
    :param filter_width: for the resblock convs
    :param residual_channels: 1x1 conv output channels
    :param dilation_channels: gate and filter output channels
    :param skip_channels: channels before the final output
    :param initial_filter_width: for the pre processing conv
    """

    @parametrized
    def wavenet(inputs,
                pre=Conv1D(residual_channels, (initial_filter_width,)),
                net=Sequential(*(ResBlock(dilation_channels, residual_channels,
                                          filter_width, dilation, out_width)
                                 for dilation in dilations)),
                post=Sequential(relu, Conv1D(skip_channels, (1,)),
                                relu, Conv1D(3 * nr_mix, (1,)))):
        inputs = pre(inputs)
        initial = np.zeros((inputs.shape[0], out_width, residual_channels),
                           'float32')
        _, out = net((inputs, initial))
        return post(out)

    return wavenet
