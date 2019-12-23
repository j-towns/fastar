# Fastar [![Build Status](https://travis-ci.org/j-towns/fastar.svg?branch=master)](https://travis-ci.org/j-towns/fastar)

Fast autoregressive sampling in the style of [fast wavenet](https://github.com/tomlepaine/fast-wavenet) and [fast pixelcnn](https://github.com/PrajitR/fast-pixel-cnn). 

Fastar has a one-function API, `fastar.accelerate`. 
To use `accelerate`, you have to express your autoregressive sampling loop as a fixed point iteration `fp` and call `accelerate(fp)`. 
Fastar accelerates the autoregressive loop using caching and allows you to `jit` sections of the accelerated loop. 
You can tune the length of jitted sections with `accelerate`'s `jit_every` kwarg, trading off between compile time and execution time.

## Example 
[PixelCNN++ sampling in colab](https://colab.research.google.com/drive/1NDvCvgqqxM0aflqzWweNtJ5P8qxrf1kr).
