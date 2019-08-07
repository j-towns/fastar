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
@click.option('--model_dir', default='./log/model', type=click.Path())
@click.option('--test_batch_size', default=128)
@click.option('--nr_filters', default=160)
@click.option('--nr_resnet', default=6)
def main(batch_size, epochs, step_size, decay_rate, model_dir, test_batch_size,
         **model_kwargs):
    t0 = time.time()
    tf.random.set_random_seed(0)
    rng = random.PRNGKey(0)
    cifar = tfds.load('cifar10')

    def train_batches():
        return tfds.as_numpy(
            cifar['train'].map(lambda el: el['image']).shuffle(1000)
            .batch(batch_size).prefetch(1))
    test_batches = tfds.as_numpy(
        cifar['test'].map(lambda el: el['image']).repeat().shuffle(1000)
        .batch(test_batch_size).prefetch(1))

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
    init_params = nn.init_fun(vmap(pcnn, (0, 0)), rng_init_1,
                              rng_init_2, next(test_batches))
    opt_state = opt_init(init_params)

    run_name = time.strftime('%Y%m%d-%H%M%S')
    for epoch in range(epochs):
        model_dir.mkdir(parents=True, exist_ok=True)
        with open(str(model_dir / f"{run_name}.npy"), 'xb') as file:
            pickle.dump(opt_get_params(opt_state), file)
        print(f"Saved model after {epoch} epochs.")

        for i, batch in enumerate(train_batches()):
            if i % 100 == 0 or i < 10:
                rng, rng_test = random.split(rng)
                test_loss = loss(opt_get_params(opt_state), rng_test, batch)
                print(f"Epoch {epoch}, iteration {i}, "
                      f"test loss {test_loss} ({time.time() - t0:.2f}s)")

            rng, rng_update = random.split(rng)
            opt_state, train_loss = update(i, opt_state, rng_update, batch)
            print(f"Epoch {epoch}, iteration {i}, "
                  f"train loss {train_loss} ({time.time() - t0:.2f}s)")


if __name__ == '__main__':
    main()
