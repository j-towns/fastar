"""
Implementation of mixture of logistic distributions in numpy
"""
import jax.numpy as np
from jax.scipy.special import logsumexp
from jax.scipy.special import expit as sigmoid
import jax.random as random


def softplus(x):
    return np.logaddexp(0, x)


def log_prob_from_logits(x, axis):
    """numerically stable log_softmax implementation that prevents overflow"""
    x = x - np.max(x, axis=axis, keepdims=True)
    return x - logsumexp(x, axis=axis, keepdims=True)


def one_hot(x, depth, dtype=np.float32):
  """Create a one-hot encoding of x of size k."""
  return np.array(x[:, np.newaxis] == np.arange(depth), dtype)


def discretized_mix_logistic_loss(theta, y, num_class=256, log_scale_min=-7.):
    """
    Discretized mixture of logistic distributions loss
    :param theta: B x T x 3 * nr_mix
    :param y:  B x T x 1
    :param num_class:
    :param log_scale_min:
    :param reduce:
    :return: loss
    """
    theta_shape = theta.shape

    nr_mix = theta_shape[2] // 3

    # unpack parameters
    means = theta[:, :, :nr_mix]
    log_scales = np.maximum(theta[:, :, nr_mix:2 * nr_mix], log_scale_min)
    logit_probs = theta[:, :, nr_mix * 2:nr_mix * 3]

    # B x T x 1 => B x T x nr_mix
    y = np.broadcast_to(y, y.shape[:-1] + (nr_mix,))

    centered_y = y - means
    inv_stdv = np.exp(-log_scales)
    plus_in = inv_stdv * (centered_y + 1. / (num_class - 1))
    cdf_plus = sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_class - 1))
    cdf_min = sigmoid(min_in)

    log_cdf_plus = plus_in - softplus(plus_in)  # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = - softplus(min_in)  # log probability for edge case of 255 (before scaling)

    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_y

    log_pdf_mid = mid_in - log_scales - 2. * softplus(mid_in)

    log_probs = np.where(y < -0.999, log_cdf_plus,
                         np.where(y > 0.999, log_one_minus_cdf_min,
                                  np.where(cdf_delta > 1e-5,
                                           np.log(np.maximum(cdf_delta, 1e-12)),
                                           log_pdf_mid - np.log((num_class - 1) / 2))))

    log_probs = log_probs + log_prob_from_logits(logit_probs, -1)
    return -np.sum(logsumexp(log_probs, axis=-1), axis=-1)


def sample_from_discretized_mix_logistic(y, rng_key, log_scale_min=float(np.log(1e-14))):
    """
    :param y: B x T x C
    :param rng_key: must be split to get new random numbers
    :param log_scale_min:
    :return: [-1, 1]
    """

    rng_key, key1 = random.split(rng_key)
    y_shape = y.get_shape().as_list()

    assert len(y_shape) == 3
    assert y_shape[2] % 3 == 0
    nr_mix = y_shape[2] // 3

    logit_probs = y[:, :, :nr_mix]

    sel = one_hot(
        np.argmax(
            logit_probs - np.log(-np.log(
                random.uniform(rng_key, np.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5))), axis=2),
        depth=nr_mix)

    means = np.sum(y[:, :, nr_mix:nr_mix * 2] * sel, axis=2)

    log_scales = np.max(np.sum(y[:, :, nr_mix * 2:nr_mix * 3] * sel, axis=2), log_scale_min)

    u = random.uniform(key1, np.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + np.exp(log_scales) * (np.log(u) - np.log(1. - u))

    x = np.min(np.max(x, -1.), 1.)
    return x
