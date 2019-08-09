from functools import partial
import pickle
import time
from pathlib import Path

import click

import tensorflow as tf
import tensorflow_datasets as tfds

from jax import jit, vmap, value_and_grad, curry
from jax import random
from jax.experimental import optimizers
import jax.numpy as np

import model
import nn


# Setup command line argument parsing
@click.command()
@click.option('--batch_size', default=32)
@click.option('--epochs', default=10)
@click.option('--step_size', default=.001)
@click.option('--decay_rate', default=.999995)
@click.option('--model_dir', default='./log/model')
@click.option('--test_batch_size', default=16)
@click.option('--nr_filters', default=160)
@click.option('--nr_resnet', default=6)
@click.option('--run_name', default=time.strftime('%Y%m%d-%H%M%S'))
@click.option('--test/--train', default=False)
def main(batch_size, epochs, step_size, decay_rate, model_dir, test_batch_size,
         run_name, test, **model_kwargs):

    print('\n'.join(f'{k}: {v}' for k, v in
                    click.get_current_context().params.items()))

    t0 = time.time()
    model_dir = Path(model_dir)
    model_file = model_dir / f'{run_name}.npy'
    tf.random.set_random_seed(0)
    rng = random.PRNGKey(0)
    cifar = tfds.load('cifar10')

    def train_batches():
        return tfds.as_numpy(
            cifar['train'].map(lambda el: el['image']).shuffle(1000)
            .batch(batch_size).prefetch(1))

    def test_batches(shuffle_and_repeat=False):
        l = cifar['test'].map(lambda el: el['image'])

        if shuffle_and_repeat:
            l = l.repeat().shuffle(1000)

        return tfds.as_numpy(l.batch(test_batch_size).prefetch(1))

    @jit
    def loss(params, rng, batch):
        batch_size = batch.shape[0]
        loss_ = vmap(partial(model.loss, pcnn), (None, 0, 0))
        losses = loss_(params, random.split(rng, batch_size), batch)
        assert losses.shape == (batch_size,)
        return np.mean(losses)

    @jit
    def update(i, opt_state, rng, batch):
        params = opt_get_params(opt_state)
        loss_val, loss_grad = value_and_grad(loss)(params, rng, batch)
        return opt_update(i, loss_grad, opt_state), loss_val

    opt_init, opt_update, opt_get_params = optimizers.adam(
        optimizers.exponential_decay(
            step_size=step_size, decay_steps=1, decay_rate=decay_rate))

    pcnn = model.PixelCNNPP(**model_kwargs)

    rng, rng_init_1, rng_init_2 = random.split(rng, 3)
    rng_init_2 = random.split(rng_init_2, test_batch_size)

    if test:
        with model_file.open('rb') as file:
            params = pickle.load(file)

        total_loss = 0.
        total_count = 0
        for i, batch in enumerate(test_batches(shuffle_and_repeat=False)):
            rng, rng_test = random.split(rng)
            test_loss = loss(params, rng_test, batch)
            print(f"Batch {i}, test loss {test_loss:.3f}")
            total_count += len(batch)
            total_loss += test_loss * len(batch)

        print(f"Overall test loss: {total_loss / total_count:.4f}")
        return

    test_batches = test_batches(shuffle_and_repeat=True)
    init_params = nn.init_fun(vmap(pcnn, (0, 0)), rng_init_1,
                              rng_init_2, next(test_batches))
    opt_state = opt_init(init_params)

    for epoch in range(epochs):
        model_dir.mkdir(parents=True, exist_ok=True)
        with model_file.open('wb') as file:
            pickle.dump(opt_get_params(opt_state), file)
        print(f"Saved model after {epoch} epochs.")

        for i, batch in enumerate(train_batches()):
            rng, rng_update = random.split(rng)
            opt_state, train_loss = update(i, opt_state, rng_update, batch)

            if i % 100 == 0 or i < 10:
                rng, rng_test = random.split(rng)
                test_loss = loss(opt_get_params(opt_state), rng_test, next(test_batches))
                print(f"Epoch {epoch}, iteration {i}, "
                      f"train loss {train_loss:.3f}, "
                      f"test loss {test_loss:.3f} "
                      f"({time.time() - t0:.2f}s)")

if __name__ == '__main__':
    main()
