import numpy as np
from jax import random

from .model import ResBlock, Wavenet, calculate_receptive_field

rng = random.PRNGKey(0)


def test_resblock(output_width=3, residual_channels=3):
    res_block = ResBlock(dilation_channels=2, residual_channels=3,
                         filter_width=5, dilation=1, output_width=3)

    inputs = np.zeros((2, 10, 1))
    initial = np.zeros((inputs.shape[0], output_width, residual_channels),
                       'float32')

    res_params = res_block.init_params(rng, (inputs, initial))
    hidden, out = res_block(res_params, (inputs, initial))
    assert hidden.shape == (2, 6, 3)
    assert out.shape == (2, 3, 3)


def test_wavenet(out_width=40):
    wavenet = Wavenet(
        dilations=[1, 2, 4], filter_width=5, initial_filter_width=2,
        out_width=out_width, residual_channels=3, dilation_channels=2,
        skip_channels=2, nr_mix=8)

    x = np.zeros((2, 100, 1))
    params = wavenet.init_params(rng, x)
    out = wavenet(params, x)
    assert out.shape == (2, out_width, 24)


def test_calculate_receptive_field():
    assert 30 == calculate_receptive_field(
        filter_width=5, dilations=[1, 2, 4], initial_filter_width=2)