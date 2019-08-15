import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path

from jax.experimental import optimizers
from jax import jit, value_and_grad, curry, np, random

import numpy as onp
from wlib.audio_reader import vctk
from wlib.mixture import discretized_mix_logistic_loss
from wlib.model import Wavenet, calculate_receptive_field


# defaults
BATCH_SIZE = 1
DATA_DIRECTORY = './data'
CHECKPOINT_EVERY = 100
NUM_STEPS = int(1e5)
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 1000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0.001
SEED = 0


def get_arguments():
    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY,
                        help='The directory containing the VCTK corpus.')
    parser.add_argument('--model_dir', type=str, default='./log/models',
                        help='Directory in which to restore the model from. ')
    parser.add_argument('--restore_file', default=None,
                        help='File name relative to model_dir for model to restore')
    parser.add_argument('--checkpoint_every', type=int,
                        default=CHECKPOINT_EVERY,
                        help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,
                        help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,
                        help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,
                        help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE,
                        help='Concatenate and cut audio samples to this many '
                        'samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,
                        default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. '
                        'Default: False')
    parser.add_argument('--nr_mix', type=int,
                        default=10,
                        help='Number of logistic mixtures'
                             'Default: 10')
    parser.add_argument('--silence_threshold', type=float,
                        default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start '
                        'and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--no_jit', action='store_true', help='Disable jit optimization for jax.')
    return parser.parse_args()


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

    def save(params):
        model_dir().mkdir(parents=True, exist_ok=True)
        with open(str(model_dir() / f'{STARTED_DATESTRING}.npy'), 'wb') as file:
            pickle.dump(params, file)

    receptive_field = calculate_receptive_field(wavenet_params["filter_width"],
                                                wavenet_params["dilations"],
                                                wavenet_params["scalar_input"],
                                                wavenet_params["initial_filter_width"])
    data_generator = vctk(args.data_dir, receptive_field,
                          args.sample_size, args.silence_threshold,
                          args.batch_size)
    print('Made data generator')

    init_batch = next(data_generator())
    print('Data shape: ')
    print(init_batch.shape)

    seq_len = init_batch.shape[1]
    output_width = seq_len - receptive_field + 1

    w_init, w_apply = Wavenet(wavenet_params["dilations"],
                              wavenet_params["filter_width"],
                              wavenet_params["initial_filter_width"],
                              output_width,
                              wavenet_params["residual_channels"],
                              wavenet_params["dilation_channels"],
                              wavenet_params["skip_channels"],
                              args.nr_mix)

    rng = random.PRNGKey(args.seed)
    opt_init, opt_update, get_params = optimizers.adam(optimizers.exponential_decay(
        step_size=args.learning_rate, decay_steps=1, decay_rate=args.lr_decay))

    restored_opt_state = restored_params_or_null()
    if restored_opt_state:
        print('Restored opt_state')
        opt_state = restored_opt_state
    else:
        init_params = w_init(rng, init_batch.shape)[1]
        opt_state = opt_init(init_params)

    print('Initialised network')

    def loss(params, batch, rng):
        # TODO: add L2 regularisation
        theta = w_apply(params, batch, rng=rng)[:, :-1, :]
        # now slice the padding off the batch
        sliced_batch = batch[:, receptive_field:, :]
        return (np.mean(discretized_mix_logistic_loss(theta,
                                                      sliced_batch,
                                                      num_class=1 << 16), axis=0)
                * np.log2(np.e) / (output_width - 1))

    @jit_with_handler
    def update(i, opt_state, batch, rng):
        params = get_params(opt_state)
        train_loss, gradient = value_and_grad(loss)(params, batch, rng)
        return opt_update(i, gradient, opt_state), train_loss

    def save_and_print(i, ls, params):
        print('Saving..')
        save(params)
        print(f'Saved model after {i} iterations.')
        onp.savetxt(str(model_dir() / f'{STARTED_DATESTRING}_losses'), ls)

    epoch = 0
    step = 0
    losses = []
    stop = False
    while not stop:
        data_iterator = data_generator()
        # go through the dataset until we have reached the step count
        print('\nStarting epoch: {}'.format(epoch))
        for batch in data_iterator:
            if step > args.num_steps:
                stop = True
                break
            opt_state, t_loss = update(step, opt_state, batch, rng)
            losses.append(t_loss)
            print('Iteration: {}, loss: {:.2f}'.format(step, t_loss))

            if step % args.checkpoint_every == 0 and step > 0:
                save_and_print(step, losses, get_params(opt_state))
            step += 1
        epoch += 1
    save_and_print(step, np.array(losses), get_params(opt_state))


if __name__ == '__main__':
    main()
