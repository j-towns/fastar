from jax.util import partial
from jax import random
import jax.numpy as np
import nn


def PixelCNNPP(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, **resnet_kwargs):
    resnet = partial(nn.gated_resnet, **resnet_kwargs)
    resnet_down = partial(resnet, conv=nn.down_shifted_conv)
    resnet_down_right = partial(resnet, conv=nn.down_right_shifted_conv)

    conv_down = partial(nn.down_shifted_conv, out_chan=nr_filters)
    conv_down_right = partial(nn.down_right_shifted_conv, out_chan=nr_filters)

    halve_down = partial(conv_down, strides=[2, 2])
    halve_down_right = partial(conv_down_right, strides=[2, 2])

    double_down = partial(nn.down_shifted_conv_transpose,
                          out_chan=nr_filters, strides=[2, 2])
    double_down_right = partial(nn.down_right_shifted_conv_transpose,
                                out_chan=nr_filters, strides=[2, 2])

    def pixel_cnn(rng, image):
        # ////////// up pass through pixelCNN ////////
        h, w, _ = image.shape
        image = np.concatenate((image, np.ones((h, w, 1))), -1)

        us  = [nn.down_shift(conv_down(image, filter_shape=[2, 3]))]
        uls = [nn.down_shift(conv_down(image, filter_shape=[1, 3]))
               + nn.right_shift(conv_down_right(image, filter_shape=[2, 1]))]

        for _ in range(nr_resnet):
            rng, rng_d, rng_dr = random.split(rng, 3)
            us.append(resnet_down(rng_d, us[-1]))
            uls.append(resnet_down_right(rng_dr, uls[-1], us[-1]))

        us.append(halve_down(us[-1]))
        uls.append(halve_down_right(uls[-1]))

        for _ in range(nr_resnet):
            rng, rng_d, rng_dr = random.split(rng, 3)
            us.append(resnet_down(rng_d, us[-1]))
            uls.append(resnet_down_right(rng_dr, uls[-1], us[-1]))

        us.append(halve_down(us[-1]))
        uls.append(halve_down_right(uls[-1]))

        for _ in range(nr_resnet):
            rng, rng_d, rng_dr = random.split(rng, 3)
            us.append(resnet_down(rng_d, us[-1]))
            uls.append(resnet_down_right(rng_dr, uls[-1], us[-1]))

        # /////// down pass ////////
        u = us.pop()
        ul = uls.pop()

        for _ in range(nr_resnet):
            rng, rng_d, rng_dr = random.split(rng, 3)
            u = resnet_down(rng_d, u, us.pop())
            ul = resnet_down_right(rng_dr, ul, np.concatenate((u, uls.pop()), -1))

        u = double_down(u)
        ul = double_down_right(ul)

        for _ in range(nr_resnet + 1):
            rng, rng_d, rng_dr = random.split(rng, 3)
            u = resnet_down(rng_d, u, us.pop())
            ul = resnet_down_right(rng_dr, ul, np.concatenate((u, uls.pop()), -1))

        u = double_down(u)
        ul = double_down_right(ul)

        for _ in range(nr_resnet + 1):
            rng, rng_d, rng_dr = random.split(rng, 3)
            u = resnet_down(rng_d, u, us.pop())
            ul = resnet_down_right(rng_dr, ul, np.concatenate((u, uls.pop()), -1))

        assert len(us) == 0
        assert len(uls) == 0

        return nn.NIN(10 * nr_logistic_mix)(nn.elu(ul))
    return pixel_cnn


def loss(pcnn, params, rng, image):
    image = nn.centre(image)
    pcnn_out = nn.apply_fun(pcnn, params, rng, image)
    conditional_params = nn.pcnn_out_to_conditional_params(image, pcnn_out)
    return -(nn.conditional_params_to_logprob(image, conditional_params)
             * np.log2(np.e) / image.size)

def sample_fp(pcnn, params, rng):
    rng_pcnn, rng_sample = random.split(rng)
    def fixed_point(image):
        image = nn.centre(image)
        pcnn_out = nn.apply_fun(pcnn, params, rng_pcnn, image)
        conditional_params = nn.pcnn_out_to_conditional_params(image, pcnn_out)
        return nn.uncentre(
            nn.conditional_params_to_sample(rng_sample, conditional_params))
    return fixed_point
