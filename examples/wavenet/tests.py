import numpy as np
from jax import random

from wlib.model import resblock_layer, Wavenet

rng = random.PRNGKey(0)

def test_resblock():
    x = np.zeros((2, 10, 1))
    dil_channels = 2
    res_channels = 3
    filter_width = 5
    dilation = 1
    out_width = 3
    res_init, res_apply = resblock_layer(dil_channels, res_channels,
                                         filter_width, dilation, out_width)
    _, res_params = res_init(rng, x.shape)
    z = res_apply(res_params, x)
    assert z[0].shape == (2, 6, 3)
    assert z[1].shape == (2, 3, 3)


def test_wavenet():
    x = np.zeros((2, 100, 1))
    dil_channels = 2
    res_channels = 3
    skip_channels = 2
    filter_width = 5
    initial_filter_width = 2
    dilations = [1, 2, 4]
    out_width = 40
    nr_mix = 8

    w_init, w_apply = Wavenet(dilations, filter_width, initial_filter_width,
                              out_width, res_channels, dil_channels,
                              skip_channels, nr_mix)

    _, w_params = w_init(rng, x.shape)
    out = w_apply(w_params, x)
    assert out.shape == (2, 39, 24)  # TODO: should this fail? shoddy test
