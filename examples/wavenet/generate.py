from __future__ import division
from __future__ import print_function

import argparse
import json
import pickle
from pathlib import Path
from collections import deque
from functools import partial
import time

from jax.experimental import optimizers
from jax.scipy.special import logit
from jax import jit, value_and_grad, curry, pmap, \
    device_count, np, random, grad

import numpy as onp
from lib.model import WaveNetModel
from lib.model_jax import Wavenet
import librosa

rng = random.PRNGKey(42)

NUM_SAMPLES = 10
TEMPERATURE = 1.0
LOGDIR = './logdir'
WAVENET_PARAMS = './wavenet_params.json'
SAVE_EVERY = None
NR_MIX = 3
SILENCE_THRESHOLD = 0.1


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a '
                             'boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError(
                    'Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument(
        '--nr_mix',
        type=int,
        default=NR_MIX,
        help='number of mixture components in output of model'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=NUM_SAMPLES,
        help='width of a single frame'
    )
    parser.add_argument(
        '--logdir',
        type=str,
        default=LOGDIR,
        help='Directory in which to store the logging '
        'information for TensorBoard.')
    parser.add_argument(
        '--restore_file',
        type=str,
        default=None,
        help='Directory in which to store the checkpointed params '
        'information for TensorBoard.')
    parser.add_argument(
        '--wavenet_params',
        type=str,
        default=WAVENET_PARAMS,
        help='file with the network parameters')
    parser.add_argument(
        '--wav_out_path',
        type=str,
        default=None,
        help='Path to output wav file')
    parser.add_argument(
        '--wav_seed',
        type=str,
        default=None,
        help='The wav file to start generation from')

    arguments = parser.parse_args()

    return arguments


def logistic_mix_sample(theta, rng, log_scale_min=-7., num_class=256):
    """
    theta has shape (num_batches, num_time_steps, 3 * nr_mix)
    """
    theta_shape = theta.shape
    nr_mix = theta_shape[2] // 3

    # unpack parameters
    means = theta[:, :, :nr_mix]
    log_scales = np.maximum(theta[:, :, nr_mix:2 * nr_mix], log_scale_min)
    inv_scales = np.exp(-log_scales)
    logit_probs = theta[:, :, nr_mix * 2:nr_mix * 3]

    def one_hot(x, dtype=np.float32):
        """Create a one-hot encoding of x of size k."""
        shape = ((1,) * (x.ndim-1) + (nr_mix,))
        return np.array(x[..., np.newaxis] == np.arange(nr_mix).reshape(shape), dtype)

    # Use the Gumbel max to sample a mixture indicator
    rng, mix_rng = random.split(rng)
    mixs = one_hot(np.argmax(
        logit_probs - random.gumbel(mix_rng, logit_probs.shape), axis=-1))
    mean = np.sum(means * mixs, axis=-1)
    inv_scale = np.sum(inv_scales * mixs, axis=-1)
    scaled_samp = inv_scale * logit(random.uniform(rng, mean.shape)) + mean
    unscaled_samp = (num_class/2 * scaled_samp) + num_class/2
    rounded = np.round(unscaled_samp)
    rounded = np.where(rounded < 0, 0, rounded)
    rounded = np.where(rounded > num_class, num_class - 1, rounded)
    return rounded

def save_wav(waveform, sample_rate, filename):
    librosa.output.write_wav(filename, waveform, sample_rate)
    print('Updated wav file at {}'.format(filename))

def main():
    args = get_arguments()

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    @curry
    def jit_with_handler(fun, *a, **kwargs):
        if args.no_jit:
            return fun(*a, **kwargs)

        try:
            return jit(fun)(*a, **kwargs)
        except Exception as e:
            print(str(e))
            return fun(*a, **kwargs)

    def model_dir():
        return Path(args.model_dir)

    def restored_params_or_null():
        if not args.restore_file:
            return None
        with open(str(model_dir() / args.restore_file), 'rb') as file:
            return pickle.load(file)

    receptive_field = WaveNetModel.calculate_receptive_field(
        wavenet_params["filter_width"],
        wavenet_params["dilations"],
        wavenet_params["scalar_input"],
        wavenet_params["initial_filter_width"])

    w_init, w_apply = Wavenet(wavenet_params["dilations"],
                              wavenet_params["filter_width"],
                              wavenet_params["initial_filter_width"],
                              1,
                              wavenet_params["residual_channels"],
                              wavenet_params["dilation_channels"],
                              wavenet_params["skip_channels"],
                              args.nr_mix)

    restored_opt_state = restored_params_or_null()

    if restored_opt_state:
        print('Restored opt_state')
        params = optimizers.get_params(restored_opt_state)
    else:
        _, params = w_init(rng, (1, receptive_field, 1))

    w_apply = jit(partial(w_apply, params))

    samples = []
    input_ = np.zeros((1, receptive_field, 1))
    for _ in range(args.num_samples):
        out_theta = w_apply(input_)
        sample = np.squeeze(logistic_mix_sample(out_theta, rng))
        samples.append(sample)
        input_ = np.concatenate([sample.reshape((-1, 1,1)), input_[:, :-1, :]], axis=1)

    samples = onp.array(samples)
    save_wav(samples, 44000, 'test.wav')


if __name__ == "__main__":
    main()
